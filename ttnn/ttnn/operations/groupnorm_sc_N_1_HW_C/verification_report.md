# Verification Report: groupnorm_sc_N_1_HW_C

Verified on a Blackhole board (11×10 = 110 worker cores). Golden suite = `eval/golden_tests/groupnorm_sc_N_1_HW_C/` (67 INPUTS shapes × the TARGET cartesian, 14 508 collected cells), acceptance suite = `tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C/` (54 cases + 5 precision-baseline rows). Three golden runs were made: **p0** (implementer's Phase 0 as handed over), **p1** (after the correctness fixes and the SUPPORTED widening), **p2** (final, after the writer store-block change and the affine pad-lane fix).

## Code Review

Every item below was **fixed in the code**; the last subsection lists what was deferred to refinements and why.

### Correctness fixes

1. **CB-sync race in pass 2 (`compute.cpp`, `build_affine_block`)** — the `b_T` chain read `a_T` (tile `tl` of `cb_a_full`) with `WaitPolicy::None` immediately after the `a_T` chain packed that tile. Unpack/math/pack are separate threads and only CB credit (`cb_wait_front`) orders "pack wrote the tile" against "unpack reads it". With `gamma_beta` the intervening `unary_bcast` of beta masked it; with `gamma_only` on RM input it surfaced as **non-deterministic per-element errors** in a few channels (golden `1x1x128x1024 RM gamma_only` rms 0.0212 > 0.02, and every top-rms cell of p0 was the `RM × gamma_only` combination at 0.010–0.021 vs 0.0017 elsewhere). Fix: `cb_wait_front(cb_a_full, tl + 1)` before the `b_T` chain. Result: rms 0.0018 and bit-deterministic across 4 runs on 4 shapes (probes 008/010). Zero device-time cost (p0 → p1 within ±2 % on every shape).
2. **Affine pad lanes were uninitialised DRAM bytes (`reader.cpp`, `fill_affine_rows`)** — for `c_non_aligned` the RM affine reader fetches a full 32-lane run of the stick, so lanes `≥ C` of the last channel tile carried whatever followed the stick. Pass 2 then writes `y = 0·a_T + β_garbage` into the output's pad lanes; harmless for bf16/fp32 output (sliced away) but a **bf8b output tile shares one exponent per 16-lane block**, so garbage there crushed the valid neighbours (p1: `1x1x64x50 bf8b gamma_beta RM-affine` PCC 0.957, `1x1x128x100` PCC 0.983; `1x1x64x17` PCC 0.996 barely passing). Fix: zero lanes `[C % 32, 32)` of the last tile's row 0 after the read. Result: PCC 0.99997 / rms 0.008, deterministic.
3. **bfloat8_b affine tiles (`reader.cpp`, host)** — a block-float tile has no addressable row 0 (shared-exponent header + packed mantissas), so the two 16-lane row-0 reads were meaningless for bf8b weights; added a `g_is_bfp` CT flag and read the whole 1088 B page (rows 1..31 are the tensor's own zero padding). Host: `Tensor.element_size()` raises for block formats — replaced with a dtype map (`_ELEM_SIZE`), only the RM stick arithmetic uses it and bf8b never takes the RM path.
4. **`compute_kernel_config` honoured per the precision convention (`groupnorm_sc_N_1_HW_C.py`)** — `validate()` now reads the caller's config (`default_compute_kernel_config()` when `None`) and refuses `fp32_dest_acc_en=False` with `ExcludedCell` for every dtype: the statistics path is a Float32-page matmul whose correctness contract is HiFi4 + fp32 DEST (`matmul_block_helpers.hpp` → Precision), and every intermediate page is `Float32`. `math_fidelity` and `dst_full_sync_en` are honoured (probe: HiFi2 and half-sync DEST both PCC 1.0000; `_dest_limit` mirrors `dest_helpers.hpp`). 16-bit DEST for bf16/bf8b is Refinement 3.

### Design-conformance / performance fix

5. **Dribbling writer (`writer.cpp`, host)** — the store loop waited and barriered on `cols` tiles at a time with `cb_out = 2·cols`. On this grid the 2-D split gives `Ct_core = 1` for every SD/SDXL shape (e.g. `(1,1,16384,320)` → `Pr × Pc = 16 × 5`, `cols = 1`), so the writer issued **one 2 KB write per NoC barrier** with a 2-tile ring — the half-done dataflow the design forbids. The store block is now an explicit host knob (`OUT_BLOCK_TILES_TARGET = 8`, `out_block = largest divisor of chunk ≤ 8`, CT arg), `cb_out = OUT_DEPTH_FACTOR · out_block`, and the writer drains `out_block` tiles per barrier walking the chunk's row-major tile order. Ledger updated. Device-time effect on the flagship shape: see Performance baseline (p1 → p2).

### Registry / contract fixes

6. **SUPPORTED under-claimed** — the kernel is dtype-agnostic (page formats come from the tensors, all statistics are Float32) and the alignment paths fall out of zero padding + the membership matrix's zero columns. Probes (010/011/012, 3–4 shapes each, several affine modes) then the full golden run confirmed: `float32` activations PCC 1.0000 / rms ≤ 0.0013; `bfloat8_b` activations PCC ≥ 0.99995 / rms ≤ 0.0098; fp32 and bf8b weights on every activation dtype; `hw_non_aligned` and `c_non_aligned` on TILE input and `c_non_aligned` on RM input at rms ≈ 0.0018. Widened `SUPPORTED["dtype"]`, `["affine_dtype"]`, `["alignment"]` accordingly. **`ROW_MAJOR × hw_non_aligned` fails structurally** (NaN on `(1,1,17,64)`, wrong statistics on `(1,1,50,128)` / `(2,1,100,128)`: `read_sticks_for_tilize` reads `32·chunk_rows` sticks per block, so the last block walks past the image's rows) → `EXCLUSIONS`, lifted by Refinement 2.

### Deferred to refinements (architectural)

- **Divisor-constrained core split** (`Pr | Ht`, `Pc | Ct` — implementer deviation from the design's ceil/floor-balanced split): 80 of 110 cores on `(1,1,16384,320)`, `(1,1,4096,320)`, `(1,1,1024,640)`, `(1,1,4096,640)`; 96 + 3 idle on `(1,1,1024,1920)`. A design-contract violation with a perf consequence; making blocks ragged touches host + all three kernels → **Refinement 1** (filed first).
- **RM × hw_non_aligned** and padding-independence of the alignment paths → **Refinement 2**.
- **16-bit DEST accumulation** (`fp32_dest_acc_en=False` for bf16/bf8b) → **Refinement 3**.

### Reviewed and left as is (with the reason)

- `ReduceInputPolicy::WaitUpfrontNoPop` + caller pop instead of the design's `BulkWaitBulkPop`: the helper asserts the input CB capacity to be a multiple of `Ht · DEST_AUTO_LIMIT` (`reduce_helpers_compute.inl`), which a `chunk_rows × cols` block with `cols < DEST_AUTO_LIMIT` cannot satisfy; the indexed access and the `Accumulate` reload are identical. Documented in the kernel header.
- Streaming regime uses a **separate** `cb_x_pass2` ring instead of the design's aliased ring: with one region the reader's pass-2 prefetch (which the stall-shadow analysis wants) could overwrite pass-1 chunks compute is still reducing. `+x_depth·chunk·x_page` (128 KiB at the defaults), recorded in the ledger.
- `handshake=False` **without** the design's gather-ready signal: records only touch rows `p < P_used`, the root zeroes only rows `≥ P_used`, and every rectangle core increments the gather semaphore *after* constructing its `ReceiverPipe`, so the totals flag cannot be clobbered by a late receiver constructor. The reasoning is in the writer header and is sound; it removes two multicast round trips per image. Any change to the combine (Refinement 5) must preserve that happens-before.
- `DestReuseBinary<Add, …, DEST_TO_SRCB>` in the apply instead of the design's `DEST_TO_SRCA`: the chain only re-programs srcA's format for `DEST_TO_SRCB`, and the CB operand always goes through unpacker A, so `DEST_TO_SRCA` with a Float32 `b_T` after a bf16 `x` trips the format check. Add is commutative. Noted as helper friction by the implementer.
- `matmul_block` per channel tile in `build_affine_block` (`M=2, K=Kg, N=1`): this is the design's own per-`T` realisation (Block Operation Realization #10) — a loop internal to one block operation, not a half-turned knob.
- `PROPERTIES` dict in the op file: not part of the registry template, harmless metadata; left.
- `NotImplementedError` for `Kg > DEST cap` (`num_groups > 256` at the default config) raised from the descriptor rather than `validate()`: a mechanism cap with no axis vocabulary; every INPUTS shape has `Kg = 1`. The multi-subblock path (`in1_num_subblocks`, `out_subblock_w`) is plumbed but unverified — keep the guard.

## Registry Conformance

- **`INPUT_TAGGERS`** = `{alignment, groups_alignment}`, both with the `(inputs, axes)` signature; `groups_alignment` reads the sibling `num_groups` from `axes` and mirrors `eval/golden_tests/groupnorm_sc_N_1_HW_C/axes.py`. ✓
- **`SUPPORTED`** covers every axis the kernel gates on: `dtype`, `layout`, `alignment`, `groups_alignment`, `affine`, `affine_dtype`, `affine_layout` (the `"none"` sentinel is listed and never refused). Widened by the verifier as described above; the op file explains the alignment rationale inline. ✓
- **`EXCLUSIONS`** = `[{layout: ROW_MAJOR, alignment: hw_non_aligned}]` with the kernel-level reason and the lifting refinement named. ✓
- **`validate()`** checks SUPPORTED per axis then EXCLUSIONS, raising `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`; it is the first statement of the public entry point, before `_validate_arguments` (ValueError contract) and any kernel work; it now also takes `compute_kernel_config` and applies the precision-convention refusal. ✓
- The op file declares **no `INVALID`**. ✓
- **INVALID audit** (`feature_spec.py`): `{dtype: bf8b, layout: ROW_MAJOR}` and `{affine_dtype: bf8b, affine_layout: ROW_MAJOR}` are each single-tensor couplings of a block format with a row-major layout (universe-must-change ✓); the eight `affine ↔ affine_dtype/affine_layout` entries are the canonical no-weight canonicalisation in both directions (presence ⇒ never the sentinel, absence ⇒ only the sentinel) ✓. No cross-tensor coupling, no "not yet implemented" entries. Nothing to change. Two suggestions for the golden-test owner (not blocking): (a) `TARGET` has no `fp32_dest_acc_en` axis although the precision convention gates on it — add it via `/golden-tests` when Refinement 3 lands; (b) `helpers._TORCH_DTYPE[bfloat8_b] = torch.bfloat16` builds the reference from bf16-rounded weights while the device consumes bf8b-quantised ones, so `float32 × bf8b-affine` cells sit at rms 0.0091–0.0093 against the fp32 target of 0.01 (without bf8b weights fp32 cells are ≤ 0.0019) — the margin is the harness's, not the kernel's; dequantising the bf8b weights for the reference (`ttnn.to_torch` of the device tensor) would remove it.

## Design Conformance

| Dimension | Verdict |
|---|---|
| Algorithm | ✓ membership-matrix (`E_T`) group masking on both the reduce side (`[S;Q] × Eᵀ`) and the expand side (`[mean;rstd] × E_T`), lane-form per-group statistics (`2·Kg` tiles), `E[x²] − mean²` in fp32 DEST with `Relu` before `+eps`, biased variance, `n_g = Cg · HW` from the host. Group-straddling and group-aligned share one path (no `(C/G) % 32` gate anywhere). |
| Data pipeline / RISC ownership | ✓ reader: x tiles / RM sticks, `E_T` generation, scaler, affine rows; compute: tilize → square → REDUCE_COL (Accumulate across chunks) → K-blocked matmul (PreKBlockFn) → root reduce → finalize chain → per-T expansion → apply; writer: gather record unicast, `Mcast2D` totals, output store. |
| Parallelisation | ✓ multi-core, one rectangle per image, 2-D `hw × ct` split, single-core-per-image for `N ≥ num_cores`, both regime pins pass. ✗ **grid fill**: divisor-constrained split leaves 73 % of the grid busy on the flagship shapes → Refinement 1 (design deviation). |
| Inter-core communication | ✓ unicast gather into the root's `cb_gather` (row `p`) + semaphore, `Mcast2D` totals broadcast via `mcast_pipe` `SenderPipe`/`ReceiverPipe`; idle rectangle cores participate in the handshake only. |
| Blocking-model fidelity | ✓ every knob is one host constant (`CHUNK_TILES_TARGET`, `X_DEPTH`, `X_RM_DEPTH`, `MEMBERSHIP_DEPTH`, `OUT_BLOCK_TILES_TARGET`, `OUT_DEPTH_FACTOR`, `MIN_TILES_PER_CORE`, `L1_BUDGET_BYTES_DEFAULT`) or derived once (`cols`, `chunk_rows`, `out_block`, `Kg`, `gather_tiles_per_stat`) and reaches the kernels as CT/RT args; kernels derive `chunk = chunk_rows · cols` and never restate a literal. No CB scales with a whole-op dimension except the predicate-guarded resident ring (the sanctioned fast-path with `streaming_2d` as the fallback). Expression: reader = one barrier per chunk ✓, compute = block-scoped helpers over `chunk` ✓, writer = was per-`cols` (→ one tile at `cols = 1`), **fixed** to `out_block`. |
| Broadcast | ✓ every broadcast operand is a row-0-valid tile with `BroadcastDim::Row` / `unary_bcast<Row>`; the gamma multiply broadcasts the row-0 gamma tile down the full `rstd_T` tile (srcB), no redundant full-tile fills of repeated data except the design's `cb_beta_full` transient (one tile). |
| Helper usage | ✓ `reduce<>` with `Accumulate`, `matmul_block<>` with `PreKBlockFn` (`InitMode::ShortAfterPreKBlock`), `tilize<>`, `eltwise_chain` / `unary_bcast` / `square`, `calculate_and_prepare_reduce_scaler`, `read_sticks_for_tilize`, `TensorAccessor` everywhere, `mcast_pipe` for the broadcast. Raw dataflow only where the design documents no helper fits (strided tile-block read/store, the P_used-way unicast gather, `E_T` generation). Kernel entry points are `void kernel_main()`, includes are `api/dataflow/dataflow_api.h`-style. |

## L1 Ledger Audit

- **Currency**: every declared CB (`0–20`, `cb_x_rm` only for RM, `cb_gamma/beta_row`, `cb_beta_full` only when present) has a row; size expressions match `create_program_descriptor`. Updated by the verifier: `cb_out` (`out_depth_factor · out_block`), `cb_scaler` (no partial pair needed — TILE padding is zero), symbol table (`out_block`, `out_block_tiles_target`, BH grid), footprint expression, dtype note.
- **Capacity vs live set**: over-capacity only where justified (streaming `x` ring depth 2, `cb_x_rm` depth 2, `cb_out` writer double-buffer); the statistic CBs, `cb_xsq`, `cb_colsum`, `cb_membership`, `cb_a/b_full` are exactly their live set. Under-capacity: none — every spanned axis (`chunk_rows`, `cols`, `Kg`, `ceil(P_max/32)`) scales its CB.
- **Page format vs DEST width**: `fp32_dest_acc_en = True` and every compute-produced page is `Float32` ✓; `cb_scaler` is a `Float16_b` dataflow-only payload ✓; `cb_x_rm`/`cb_out` carry the tensor formats ✓. The design's `cb_xsq` `Float16_b` lamp (2× fewer bytes through packer/unpacker for bf16/bf8b input) is a measured knob — folded into Refinements 3/4.
- **Disjoint lifetimes**: `cb_xsq ↔ cb_a_full + cb_b_full`, `cb_agg_interm ↔ cb_stats_T`, `cb_stats_row ↔ cb_totals_recv` are all recorded decisions with a reason (page-count mismatch / ≤ 64 KiB saving / DEST-window ordering). Nothing silent.
- **Bounds / closed form**: `cols ≤ DEST_AUTO_LIMIT` (host clamp), `chunk_rows ≤ chunk_tiles_target / cols`, `Kg ≤ DEST cap` (guarded), `P_max ≤ grid`, `blk` predicate-guarded; total is closed-form. Worked default ≈ 550 KiB streaming / resident up to ≈ 277 bf16 tiles per core — every INPUTS shape is resident at ≥ 56 cores.
- **Data-movement budget**: present and consistent (x once resident / twice streaming, y once, gamma/beta `Pr` re-reads of ≤ 128 B per tile, statistics never touch DRAM, `P_used · 256 B + 8 KiB` cross-core per image); the cheapest-traffic split is the one implemented, `mcast_affine` rejected on payload size, `all_gather_combine` a `deferred` row with a positive reason.
- **Block-size defaults**: interleaved → full-grid split first ✓ (modulo the divisor constraint → Refinement 1), then `chunk_tiles_target = 32` rather than "coarsest that fits" — a deliberate design lamp (`compute_block_size` ≈ 1.6 µs per extra pass vs `double_buffer` saturation at 4–8 tiles) to be **measured** in Refinement 4, not a settled solve. No block was shrunk to make a buffer fit.
- **Per-core footprint** (bf16, `cols = 8`, `chunk_rows = 4`, `Kg = 1`, gamma+beta): `fixed ≈ 422 KiB` (`cb_xsq 128 KiB ∝ chunk`, `cb_colsum 64 KiB + cb_a/b_full 64 KiB + cb_membership 32 KiB ∝ cols`, statistic CBs `≈ 56 KiB ∝ Kg` (+`8 KiB ∝ ceil(P_max/32)`), `cb_out 32 KiB ∝ out_block`) + resident `blk · x_page` or streaming `2 · x_depth · chunk · x_page = 256 KiB`.

## Precision Baseline

`tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C/test_groupnorm_sc_N_1_HW_C_precision_baseline.py` — bf16 in/out, TILE input, RM bf16 affine, `randn` input (seed 1234), fp32 torch reference:

| Shape | G | affine | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | bf16 ULP (mean / max) | got/true ratio p5 / median / p95 |
|-------|---|--------|-----|-------------|--------------|------------------|------------------------|----------------------------------|
| (1,1,32,32) | 1 | — | 0.999999 | 0.0065 | 0.00106 | 0.0015 | 0.30 / 25.7 | 0.9968 / 0.9998 / 1.0024 |
| (1,1,1024,640) SD1.5 | 32 | γ,β | 0.999999 | 0.0372 | 0.00164 | 0.0017 | 0.48 / 1.7e4 (near-zero refs) | 0.9972 / 1.0002 / 1.0033 |
| (4,1,128,256) | 8 | γ,β | 0.999999 | 0.0341 | 0.00162 | 0.0017 | 0.45 / 1.4e3 | 0.9971 / 1.0002 / 1.0033 |
| (1,1,16384,320) SDXL | 32 | — | 0.999999 | 0.0189 | 0.00127 | 0.0019 | 0.30 / 8.3e3 | 0.9981 / 1.0009 / 1.0036 |
| (1,1,1024,640), x = randn + 8 (offset-heavy) | 32 | γ,β | 0.99972 | 0.540 | 0.0235 | 0.028 | 22.8 / 2.0e5 | 0.8996 / 0.9952 / 1.0699 |

**Assessment**: on zero-mean data the op is at the bf16 output-rounding floor (mean error 0.3–0.5 bf16 ULP, relative RMS 0.0015–0.0019, PCC 0.999999); the got/true ratio is a broad cluster centred on 1.000 — **no uniform scale offset**, so no structural bug is hiding behind a high PCC. The **offset-heavy** row (|mean| = 8σ) degrades to relative RMS 0.028 with the ratio spread still centred on 1.0 (rounding, not scale): the apply computes `x·a_T + b_T` with `|x·a_T| ≈ |b_T| ≈ 8·|γ|` and cancels them at the FPU's tf32-class operand precision, exactly the limitation the implementer documented in `compute.cpp`. It does not affect any golden cell (all `randn`) — see Recommendations for the lever.

Golden-suite precision distribution (p2, 3 597 passing cells): bf16 rms median 0.0017 / p99 0.0087 / max 0.0097 (target 0.02), PCC ≥ 0.99996; fp32 rms median 0.0010 / max 0.0093 (target 0.01 — the 0.009 tail is entirely the `× bf8b-affine` reference artefact noted above; ≤ 0.0019 otherwise), PCC ≥ 0.99996; bf8b rms median 0.0123 / p99 0.0151 / max 0.0161 (target 0.10), PCC ≥ 0.99988 on every alignment (the p1 `(1,1,64,17)` 0.996 outlier was the pad-lane bug, now 0.9999).

**Recommended tolerances**: bf16 PCC ≥ 0.999 / rel-RMS ≤ 0.01 on zero-mean data (the harness's 0.995 / 0.02 leaves 2× margin); fp32 PCC ≥ 0.9999 / rel-RMS ≤ 0.005 with non-bf8b weights; bf8b PCC ≥ 0.999 / rel-RMS ≤ 0.03. For `comp_allclose`-style checks on bf16: `rtol = 0.02, atol = 0.05`.

## Verifier CLI Summary

Final run (**p2**, all fixes in): see the table's last column — every loud category is 0.

| Category | p0 (as handed over) | p1 (fixes + widened SUPPORTED) | **p2 (final)** |
|---|---|---|---|
| supported_pass | 519 | 3 595 | **3 597** |
| xfail_expected | 3 165 | 88 | **88** |
| invalid_skipped / no_axes_found (INVALID + collection-time skips) | 10 823 | 10 823 | **10 823** |
| supported_fail | 1 (`RM × gamma_only` CB race) | 2 (bf8b × c_non_aligned pad lanes) | **0** (must be 0 to ship) |
| xpass_drift | 0 | 0 | **0** (must be 0 to ship) |
| xfail_wrong_mode | 0 | 0 | **0** (must be 0 to ship) |
| supported_marked_xfail | 0 | 0 | **0** |
| hangs | 0 | 0 | **0** |

The 88 `xfail_expected` cells are exactly the `EXCLUSIONS` entry (`ROW_MAJOR × hw_non_aligned`: 4 shapes × {bf16, fp32} × 11 affine cells) → Refinement 2. Every other `TARGET − SUPPORTED` value is now in SUPPORTED; `bf8b × ROW_MAJOR` (activation or weight) is `INVALID`.

Artefacts: `/tmp/gn_results_p{0,1,2}/{test_results.json,verifier_report.json}` on the verification host (the final `verifier_report.json` is 7.4 MB — above the repo's 500 KB pre-commit limit — so it stays in the results dir per the pipeline convention rather than next to this file).

## Performance baseline (Phase 0, bf16 × TILE input, median `device_kernel_ns` over that shape's golden cells, 11×10 BH grid)

| Shape | µs (p1) | µs (p2, batched writer) | cores used | tiles | MB in+out | GB/s effective (p2) |
|---|---|---|---|---|---|---|
| (1,1,16384,320) SDXL | 84.9 | 85.3 | 80 / 110 | 5120 | 21.0 | 246 |
| (1,1,1024,1920) | 63.7 | 63.4 | 96 (+3 idle) | 1920 | 7.9 | 124 |
| (1,1,4096,640) | 50.9 | 51.6 | 80 | 2560 | 10.5 | 203 |
| (1,1,4096,320) | 28.3 | 29.0 | 80 | 1280 | 5.2 | 181 |
| (1,1,1024,640) SD1.5 | 21.6 | 22.0 | 80 | 640 | 2.6 | 119 |
| (1,1,256,1280) | 18.0 | 17.6 | 80 | 320 | 1.3 | 74 |
| (1,1,128,128) | 7.8 | 7.8 | 16 | 16 | 0.07 | — (latency floor) |
| (1,1,32,32) | 6.9 | 6.9 | 1 | 1 | 0.004 | — (latency floor) |

(RM-input cells of the same shapes are slower by the in-kernel tilize — e.g. SDXL RM ≈ 200 µs — which is why a mixed-layout median reads ≈ 142 µs.)

Per-dtype on SDXL (p2, TILE): bf8b 59.2 µs, bf16 85.3 µs, fp32 145.2 µs — effective bandwidth 190 / 246 / 290 GB/s, i.e. it *rises* with tile size: the reader is largely per-transaction-bound (one `noc_async_read` per tile; nothing DRAM-contiguous at `Ct_core = 1`), not DRAM-bandwidth-bound. `(1,1,1024,1920)` is the worst large shape (124 GB/s on 96 cores): `Pr = 32` leaves `Ht_core = 1`, so its chunks are `1 × 5` tiles — the `Pr`-tie-break produces thin chunks there. The writer store-block change (fix 5) is **measured neutral** at the current chunk sizes (all-cell p2/p1 ratio median 1.001, p10–p90 0.99–1.01): it restores the design's batching contract but the reads, not the writes, bound these shapes. Every shape ≤ 64 tiles sits at a 6.9–10 µs floor (launch + gather/multicast handshake) where the data is < 1 µs of DRAM time. These are the regions the perf refinements target (`op_requirements.md` Refinements 1, 4, 5).

## Recommendations

- **Queue order** (`op_requirements.md`): R1 grid fill (design deviation, perf), R2 RM × hw_non_aligned + padding independence, R3 16-bit DEST config surface, R4 block-knob co-tune incl. transfer coalescing, R5 combine latency floor. R2 and R3 are the only generality items left; everything else in TARGET is supported.
- **Offset-heavy inputs** (|mean| ≫ σ): the apply's `x·a_T + b_T` cancellation costs ~2.8 % relative RMS at |mean| = 8σ (measured above). The concrete lever is a re-associated apply — `(x − mean_T)·(rstd_T·γ_T) + β_T` (subtract first, so no O(|mean|) intermediates; one extra FPU op per tile from the already-resident `mean_T_full`) — or the design's deferred `shifted_two_pass_variance` for the *statistics* side. No golden cell fails (all `randn`), so this is not queued; revisit if a model-derived loose case with a non-zero-mean `input_gen` is added to `feature_spec.LOOSE_CASES`.
- **Thin harness margin**: `float32 × bf8b-affine` cells sit at rms 0.0091–0.0093 against the fp32 target of 0.01 — a reference-quantisation effect of the harness (`bf16` stand-in for bf8b weights), not a kernel defect; a seed change could flip them. Dequantising bf8b weights in the reference removes it.
- **Padding assumptions**: TILE-input `hw_non_aligned` / `c_non_aligned` rely on the input's tile padding being zero (true for `ttnn.from_torch`, documented in the design's Key Risks). Refinement 2 makes both layouts padding-independent (partial COL scaler for the last tile-row, explicit RM pad zero-fill).
- **L1 headroom**: the worked default leaves ≈ 450 KiB of the 1 MB budget for the resident ring; `cols·Kg·F32` (`cb_membership`) and `2·Kg·F32·(5 + ceil(P_max/32))` grow with `num_groups` — at `G = 256` (`Kg = 8`) the statistic CBs reach ≈ 400 KiB and residency shrinks to ≈ 150 bf16 tiles per core; no INPUTS shape approaches it, but a `num_groups > 256` request hits the `Kg > DEST cap` guard first.
- **Helper friction worth upstreaming** (from the implementer's breadcrumbs, confirmed during review): `BulkWaitBulkPop` REDUCE_COL capacity assert vs the documented block shape; `DestReuseBinary<…, DEST_TO_SRCA>` not reconfiguring unpacker A's format; `mcast_pipe` `ReceiverPipe` constructor resetting the flag with `handshake=False`; the Blackhole `src % 64 == dst % 64` NoC read rule for sub-tile RM reads.
