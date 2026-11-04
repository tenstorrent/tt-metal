# Verification Report: nomic_moe_combine

Verifier pass on Phase 0 (`owner_computes`, realization R2), 2026-11-04, Blackhole p300c (11x10 grid).
Golden results: `/tmp/nmc_verify/golden2/` (`verifier_report.json` copied next to this file as
`verifier_report.json`).

## Code Review

### Prompt rules (`eval/prompts/nomic_moe_combine.txt` § Rules)

| Rule | Status |
|------|--------|
| MUST be one `ttnn.generic_op` dispatch; MUST NOT sequence ttnn ops | ✓ one `generic_op`; the only other device call on the path is `ttnn.allocate_tensor_on_device` (the output) |
| MUST read index / weights on the device; never host-read, never CT args / cached state | ✓ addresses are RT args; only shape-derived values (`token_tiles`) are CT |
| A call allocates only its output; no host-built tensor | ✓ |
| MUST write every output tile, padding rows included, in memory_config's layout | ✓ packs the whole shard; padding rows = bias exactly (acceptance + golden check) |
| MUST pass io tensors as [expert_rows, combine_index, combine_weights, bias, output] | ✓ |
| Kernel names `nomic_moe_combine_{reader,compute,writer}.cpp` | ✓ reader (two instances, CT `choice`), compute; no writer (no NoC writes) |
| MUST NOT change shared sources / host C++; no builds | ✓ only the op's Python + kernels |
| *prefer* computing the per-shape plan once and reusing it | **advisory (partial):** the `Plan` (work split, core list, per-core args) is cached, but `create_program_descriptor` rebuilds the CB descriptors, CT arg lists and the 96-core `RuntimeArgs` every call, and `validate()` re-derives the norm layout (`input_memory_config`) every call. Kernel time is unaffected; the forward's host wall clock pays it 6x. The output CB is bound to the output buffer, so a cached descriptor needs an address-patch path; left as a recommendation, not a refinement (no device-ns gate). |

### Fixed in place

1. **DRY: `top_k` restated as a literal in the compute kernel.** `nomic_moe_combine_compute.cpp` declared
   `constexpr uint32_t top_k = 2;` independently of the host's `TOP_K`. It is now CT arg 12, fed from `TOP_K`, with
   `static_assert(top_k == 2)`, because the main loop is written for one rows/weights CB pair per choice.
2. **DRY: CB footprint computed twice.** `cb_bytes()`, which backs `circular_buffer_bytes()` and validate()'s 384 KB
   budget check, re-derived every CB size in a formula parallel to the descriptor's `_cb(...)` calls. Both now come
   from one list, `circular_buffer_specs(r, W, rows_depth, fused)` in the program-descriptor module. The descriptor
   allocates exactly that list and `cb_bytes` sums it, so the reported figure cannot drift from what is allocated.
   Values are unchanged (checked against the ledger's closed form for all five plan shapes).
3. **Mechanism cap not enforced on the host: transaction ids vs. shard height.** The reader gives every tile row's
   weight tile its own transaction id (`batch_depth + 3 + r <= 15`, a kernel `static_assert`). validate() only bounded
   `r` through the CB budget, which admits up to r ≈ 30 at W = 1. A plan with r > 10 would have passed validate()
   and then failed to compile. `_shard_geometry` now refuses `r > NOC_MAX_TRID - BATCH_DEPTH - SIDE_TRIDS` with a
   `SupportRefusal`. No norm plan comes close (r ≤ 4).
4. **Ledger currency** (`l1_ledger.md`). The symbol table said 4 x 2 reads in flight; the code runs 8 x 2. The trid
   bound was stated as `batch_depth + 3 ≤ 15`. The R2 totals table counted the absent `cb_weights_bcast`, so every
   row was 2·r·4096 B too high (e.g. 8x512: 225 KB, not 257 KB). All corrected, plus an audit note (below).
5. **Precision-baseline test added**: `test_nomic_moe_combine_precision_baseline.py`.

### Reviewed, no change needed

- **Registry conformance.** `INPUT_TAGGERS` (`tag_window`, `tag_batch`, both `(inputs, axes)`), `SUPPORTED` (all five
  TARGET axes), `EXCLUSIONS = []` and `validate()` are all present. validate() checks SUPPORTED per axis, then
  EXCLUSIONS, then the domain. It is the first statement of the entry point. The op file declares no `INVALID`.
- **Correctness.** CB push/wait/pop counts balance on every CB: rows W per tile row; raw weights 1 per tile row,
  waited cumulatively and popped once; bias W once; tilized 2W per tile row; output shard reserve/push per DEST
  group, never popped (it is the tensor). Kernels use `TensorAccessor`, `void kernel_main()` and the `api/...`
  include paths. Every trid barrier has a later read behind it, and the last read uses the full barrier (the
  MOE_GLUE §4 race rule).
- **Native sharding.** The output CB is the shard itself (`cb_descriptor_from_sharded_tensor`, zero copy, no writer).
  The local shard is never re-read through an accessor.
- **Helper usage.** Tilize uses `compute_kernel_lib::tilize` (InitOnly / UninitOnly across the two choices, no
  reconfig), and the R1 expansion uses `unary_bcast<Col>`. Raw compute-API calls for the DEST group (`unary_bcast<ROW>`
  seed plus `mul_tiles_bcast_cols` accumulate, inits hoisted over the group) and the stateful one-packet NoC reads
  are the design's documented helper rejections (op_design § Helpers considered and rejected). These are mechanism
  choices, not conformance findings.
- **Broadcast.** Weights use column broadcast from the raw tile (faces 0 and 2 only). The bias uses row broadcast from
  row 0 of faces 0 and 1. Neither CB is filled with replicated data.
- **Debug scaffolding** (`ABLATE_*`, `NMC_ZONE`) is compiled out unless `NOMIC_MOE_COMBINE_ABLATE` is set. I kept it;
  the ablations below depend on it.

### Design conformance

| Dimension | Verdict |
|-----------|---------|
| Algorithm | ✓ one fp32 DEST accumulation (bias seed, then k = 0 and k = 1 products), one bf16 pack. R2 (tilize to a CB, then `mul_tiles_bcast_cols`) replaces R1 as the default. This is the design's named fallback, chosen by measurement (R1 32.7 µs vs R2 16.8 µs at 8x512) |
| Pipeline topology | ✓ reader k=0 on RISCV_0/NOC_0 and k=1 on RISCV_1/NOC_1, no writer, compute packs into the shard |
| Work distribution | ✓ exactly the norm shard grid (64 / 96 / 96 cores), one shard per core. Idle cores: 46 at 1k, 14 at 2k/4k (see Recommendations: the deferral premise is weakened) |
| Inter-core communication | ✓ none (as designed) |
| Blocking-model fidelity | ✓ r, W, rows depth, batch_reads x batch_depth, DEST group and realization are all host knobs with one source each (after fixes 1–2). The whole-shard rows CB is predicate-guarded by the 384 KB budget check, which is the sanctioned residency path, not a collapsed knob |
| Expression | ✓ reader: batches of 8 reads, one push of W pages per tile row. Compute: whole tile row (W tiles, both choices) per tilize pass, DEST groups of `DEST_AUTO_LIMIT`. No per-unit handshake inside a block |

## Registry Conformance

- Confirmed: INPUT_TAGGERS, SUPPORTED, EXCLUSIONS and validate() are present and wired correctly. The op file
  declares no INVALID.
- No XPASS-driven auto-fixes (xpass_drift = 0).
- **INVALID audit** (`feature_spec.py`): `INVALID = []`. That is correct: no single-tensor structural impossibility
  exists among the TARGET axes. `dtype` has only bf16 and there is no layout axis, so the canonical
  bf8b + ROW_MAJOR entry does not apply. The op is not norm-like (no weight canonicalization). The one impossible
  tag combination, `window=ragged` x `batch=multi` (B > 1 ⇒ S % 32 == 0 ⇒ T % 32 == 0), is shape-derived. No INPUTS
  entry produces it, so it never reaches the parametrization and needs no INVALID entry.

## L1 Ledger Audit

- **Currency:** every allocated CB has a row and every row matches a CB. Size expressions now match the allocation
  (fix 4). Rows for R1-only `cb_weights_bcast` and R2-only `cb_rows_tilized` are marked by realization.
- **Capacity vs live set:** `cb_rows_choice{k}` holds the whole shard (r·W) while compute consumes one tile row at a
  time. That is justified (the reader never stalls on CB space; lamp L6 measured depth 2 as equal) and predicate-
  guarded by the budget. `cb_weights_raw_*` uses 2 KB of each 4 KB page (faces 1 and 3 unwritten), justified
  because the column-broadcast unpack needs face 2 at +2 KB. No under-sized spanned axis.
- **Page format vs DEST width** (`fp32_dest_acc_en` on): `cb_rows_tilized` is Float16_b but holds exact bf16 copies
  (tilize is lossless), so this is a valid mechanism reason. Weights and bias are Float32 dataflow-only payloads,
  matching srcB (no reconfig). The output is bf16 by contract. No findings.
- **Disjoint lifetimes:** `cb_weights_raw_*` vs `cb_weights_bcast` (R1) and `cb_rows_tilized` vs `cb_weights_bcast`
  (never both allocated) are both justified in the ledger.
- **Bounds:** r (budget and, newly, the trid cap) and W (∈ {1,2,3,4,6,8,12}, bias staging cap) are both bounded in
  validate(). The total is closed-form: R2 = `4096·r·W + 8448·r + 8192·W` B. The largest plannable footprint
  (r=4, W=12) is 321 KB, under 384 KB.
- **Block-size defaults:** the sharded default (one block = the whole resident shard) holds.
- **Data-movement budget:** present and consistent with the code (rows 12.6 MB once, index 3x, weight faces 1.5x,
  bias W·128 B a core, 0 output NoC bytes).
- **Finding: the cheapest-traffic deferral lost its positive reason.** `group_shared_index_weights` was deferred as
  "off the critical path". Measured: removing the weight reads takes 8x512 from 14.86 to 12.03 µs and 4x512 from 8.77
  to 7.88 µs. The deferral reason is therefore false, so I moved it to the queue as **Refinement 2** (perf, scheme-
  change). I recorded this in the ledger's new "Verifier audit" section rather than filing a separate ledger entry.
- Per-core footprint at the perf shape (8x512, r=4, W=8): 225 KB. The `r·W` term (rows CBs) scales with both block
  knobs; `r` terms with the token-row block; `W` terms with the hidden split.

## Precision Baseline

`test_nomic_moe_combine_precision_baseline.py`: real_text routing, rows in L1, norm layout, against the float64
reference over the window's tokens.

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP p99 / ≤1 ULP share | got/true ratio p5 / p50 / p95 |
|-------|-----|-------------|--------------|------------------|------------------------|-------------------------------|
| 1x1000 (G=2, r=1, ragged) | 0.9999986 | 8.14e-3 | 6.60e-4 | 0.00167 | 2 / 98.79% | 0.99701 / 0.99991 / 1.00282 |
| 2x512 (G=2, r=1) | 0.9999986 | 8.51e-3 | 6.61e-4 | 0.00168 | 2 / 98.77% | 0.99699 / 0.99991 / 1.00283 |
| 4x512 (G=3, r=2) | 0.9999986 | 8.24e-3 | 6.62e-4 | 0.00168 | 2 / 98.78% | 0.99700 / 0.99992 / 1.00283 |
| 8x512 (G=3, r=4) | 0.9999986 | 8.56e-3 | 6.62e-4 | 0.00168 | 2 / 98.77% | 0.99700 / 0.99991 / 1.00283 |

ULP is measured against the correctly rounded exact value, taking the nearer of ties-to-even and ties-away. 92% of
values are bit-exact. The ULP maximum (~29k) occurs only where the result cancels to near zero: there the TF32
weight rounding the contract allows (relative to the products, not the result) is many of the result's own ULPs.
ULP is not a meaningful gate in that regime, so the test gates on PCC and relative RMS (the golden bars) plus a
≥ 98% within-1-ULP regression guard.

**Assessment:** this matches the incumbent's calibration exactly (PCC 0.9999986, relative RMS 0.00167–0.00168): one
rounding of an fp32 accumulation with TF32 weights. The ratio is centred on 1.0 with a symmetric ±0.3% spread, which
is ordinary rounding noise; there is no scale or structural signature. The one-hot exact test, the padding = bias
test and the bias-multiplicity = 1 test all pass (acceptance + golden regression).
**Recommended tolerances:** keep the golden bars, PCC ≥ 0.999995 and relative RMS ≤ 0.0019 (about 13% headroom on
RMS). Elementwise allclose holds at rtol=1.6e-2, atol=1e-2.

## Performance baseline (device kernel duration, board AICLK)

Three runs of the LOOSE_CASES setup (real_text, 96 KB reserved, rows L1, norm layout):

| Shape | Cores | Device kernel (µs) | Accept | Stretch | Incumbent |
|-------|-------|--------------------|--------|---------|-----------|
| 8x512 (perf focus) | 96 | 14.69 / 14.79 / 14.85 | 20.0 ✓ | 13.0 ✗ | 32.22 |
| 4x512 | 96 | 8.55 / 8.60 / 8.84 | 12.5 ✓ | 8.0 ✗ | 19.33 |
| 2x512 | 64 | 6.80 / 6.90 / 7.33 | 8.5 ✓ | 5.5 ✗ | 12.88 |

**Ablation** (8x512 / 4x512 / 2x512, µs; outputs wrong by construction):

| Removed | 8x512 | 4x512 | 2x512 |
|---------|-------|-------|-------|
| nothing (baseline) | 14.86 | 8.77 | 6.89 |
| row reads | 12.78 | 7.15 | 5.43 |
| weight reads | 12.03 | 7.88 | 6.44 |
| multiply + bias seed + tilize (all compute math) | 11.33 | 6.36 | 4.34 |
| all reads (rows, weights, bias) | 9.77 | 5.74 | 4.25 |
| all reads + multiply | 5.72 | 3.71 | 2.76 |
| all reads + tilize | 7.86 | 4.92 | 3.58 |
| all reads + bias seed | 8.99 | 5.33 | 3.95 |
| all reads + all compute math (fixed floor) | 3.08 | 2.33 | 1.69 |

**Reading:** with no reads at all, 8x512 still takes 9.8 µs, so compute is co-critical with the reads (reads alone
come to ~11.3 µs). With reads removed, compute breaks down into the HiFi4 multiply ~4.0 µs (~170 cycles an output
tile for two products), tilize ~1.9 µs and the bias seed ~0.8 µs, over a ~3.1 µs fixed floor (launch, index DRAM
read on 2 banks for 192 readers, handshakes). Overlap is partial: compute idles until a tile row's 64 rows land, and
the first tile row lands late behind the grid's weight-read DRAM burst (implementer note). This sets the two perf
refinements: cut the DRAM side traffic (Refinement 2) and cut per-tile compute (Refinement 3).

## Verifier CLI Summary

- supported_pass: 28 (14 golden shapes + 3 loose perf cases + 11 regression)
- xfail_expected: 70 (rows_buffer=dram x 3 output layouts x 14 shapes = 42; rows_buffer=l1 x {l1_interleaved,
  dram_interleaved} x 14 = 28). **All 70 are covered by Refinement 1.**
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

## Recommendations

- **Queue** (`op_requirements.md`): R1 placement generality (DRAM rows + interleaved outputs); R2 perf: group-shared
  index and weights at 8x512; R3 perf: compute per-tile cost and overlap at 8x512.
- **Occupancy premise weakened.** `owner_plus_helpers` and `whole_row_remote_write` were deferred because "the read
  phase is NoC-aggregate-bound". The ablation shows compute co-critical (9.8 µs with no reads at 8x512), and 14 of
  110 cores idle at 4k (46 at 1k). If R2 and R3 together still miss the 13.0 µs stretch, the next perf phase should
  build and measure one of these two (`whole_row_remote_write` with W = 24 on 110 cores is the single experiment
  op_design names).
- **op_design.md is stale on two points** (planner artifact, not edited here): the mechanism-cap row
  "`batch_depth + 3 ≤ 15`" (actual: `batch_depth + 3 + r ≤ 15`) and the Phase 0 reads-in-flight value of 4 x 2
  (actual: 8 x 2, measured).
- **Host time per call** (soft prompt rule): cache the descriptor skeleton per plan (CBs except the output shard, CT
  args, core lists) and patch only the five addresses. Also cache validate()'s norm-layout derivation by
  `(shape, device id)`.
- **Precision:** none. There is no failing cell and the bars have headroom. `validate()` pins HiFi4 +
  `fp32_dest_acc_en` (the precision contract); R3 may probe a lower fidelity only under the unchanged golden bars.
