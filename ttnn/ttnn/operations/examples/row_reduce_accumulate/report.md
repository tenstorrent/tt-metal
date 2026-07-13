# Row-mean reduce — cross-tile accumulation methods (single core)

box=bgd-lab-t3003-special-mstaletovic-for-reservation-40918  arch=WH_B0  cores=1  placement=single-core sharded-L1  N=5 (median)  kernel-iters=200 (steady-state)
problem: mean across a row of W tiles ([32, 32*W] -> [32,1] row-means), output fp32, fidelity=HiFi4.

Axes: METHOD (how the W tiles are summed) x PRECISION '<input>-<accum>' x input DISTRIBUTION.
  methods:  reduce_fold (fold sum into the reduce, baseline) | l1_accum (packer L1-accumulate) |
            dest_accum (add_tiles acc_to_dest, 1 tile/add) | dest_accum_pairs (2 tiles/add)
  precision: fp32-fp32 (both precise) | bf16-fp32 (lossy input, precise accum) | bf16-bf16 (+accum loss)
  distributions: signal (per-row linspace+noise, large) | uniform [-1,1) | positive [0,1)
  (l1_accum's packer L1-acc is fp32-DEST-only, so its '-bf16' rounds only the L1 accumulator CB.)

## Overview  (perf: bf16 input, ns per row-mean; accuracy: bf16 ACCUMULATION vs fp64, at the widest row)

Perf = median ns at 1t (narrow) and 32t (wide, ×vs reduce_fold), bf16 input, kernel-iters=200. Accuracy = `max_abs \| max ULP_bf16` of the **bf16-bf16** config at 32t, per input distribution — where precision is actually lost (fp32 accumulation is ~exact).

| method | ns @1t | ns @32t (×) | signal: max\|ULP | uniform: max\|ULP | positive: max\|ULP |
|---|---:|---:|---:|---:|---:|
| reduce_fold | 466 | 5311 | 2.1e-01 \| 13u | 7.1e-04 \| 40u | 1.1e-02 \| 6u |
| l1_accum | 810 | 5150 (1.03x) | 3.7e-03 \| 0u | 2.1e-04 \| 12u | 8.6e-04 \| 0u |
| dest_accum | 846 | 2868 (1.85x) | 3.3e-02 \| 2u | 6.3e-04 \| 328u | 5.7e-03 \| 3u |
| dest_accum_pairs | 821 | 1806 (2.94x) | 2.1e-02 \| 1u | 3.7e-04 \| 112u | 4.5e-03 \| 2u |
| dest_accum_sfpu | 918 | 2923 (1.82x) | 2.4e-02 \| 2u | 5.4e-04 \| 319u | 4.0e-03 \| 2u |
| dest_accum_pairs_sfpu | 904 | 1856 (2.86x) | 8.2e-03 \| 1u | 2.8e-04 \| 132u | 2.0e-03 \| 1u |

How to read it:
- **fp32 accumulation is essentially exact** (fp32-fp32 and bf16-fp32 keep max_abs ≤ ~3e-3 for every method/distribution/width), so the bf16-bf16 numbers above are the whole accuracy story.
- **bf16 *input* alone is nearly free** for a wide mean: its error *averages DOWN* with width (see the bf16-fp32 detail); bf16 *accumulation* is what *grows UP* with width and separates the methods.
- On **signal / positive** (nonzero mean) the ordering is reduce_fold (worst) > dest_accum > dest_accum_pairs > l1_accum (best) — the running sum swamps small increments in bf16, worst when it is folded whole into one bf16 DEST (reduce_fold). **dest_accum_pairs is both fastest and the more accurate DEST-add method.**
- On **uniform** (zero-mean) max_abs is tiny for ALL methods (~1e-3) — a near-zero mean has little magnitude to lose — but ULP is large because it divides that error by the ~0 mean (relative error is high near zero). So max_abs, not ULP, is the honest metric for cancellation-heavy data; there the method choice barely matters in absolute terms.
- **SFPU finalize** (dest_accum_sfpu / dest_accum_pairs_sfpu) does the within-tile collapse on the SFPU in DEST (`sfpu_reduce` + a scalar-multiply for 1/N) instead of the FPU reduce library. It reads DEST natively (no pack->L1->unpack round-trip) yet is NOT faster — the SFPU vector reduce costs more than the FPU matmul-reduce, marginally outweighing the saved round-trip — but it is slightly MORE accurate in bf16 (it collapses the columns in fp32 internally before the single output rounding).
- fp32 input roughly halves the wide-row perf win (it unpacks 2× the bytes); see the per-precision perf.

## Perf — median ns per row-mean, distribution=signal (data-independent); speedup vs reduce_fold

### fp32-fp32

| method | 1t (32e) | 2t (64e) | 4t (128e) | 8t (256e) | 16t (512e) | 32t (1024e) |
|---|---|---|---|---|---|---|
| reduce_fold | 489±0% | 651±0% | 976±0% | 1594±0% | 2835±0% | 5338±0% |
| l1_accum | 846±0%  (0.58x) | 988±0%  (0.66x) | 1301±0%  (0.75x) | 1862±0%  (0.86x) | 3009±0%  (0.94x) | 5310±0%  (1.01x) |
| dest_accum | 907±0%  (0.54x) | 1029±0%  (0.63x) | 1297±0%  (0.75x) | 1818±0%  (0.88x) | 2870±0%  (0.99x) | 4965±0%  (1.08x) |
| dest_accum_pairs | 850±0%  (0.57x) | 902±0%  (0.72x) | 1020±0%  (0.96x) | 1283±0%  (1.24x) | 1811±0%  (1.57x) | 2847±0%  (1.88x) |
| dest_accum_sfpu | 993±1%  (0.49x) | 1070±0%  (0.61x) | 1355±1%  (0.72x) | 1875±0%  (0.85x) | 2924±0%  (0.97x) | 5022±0%  (1.06x) |
| dest_accum_pairs_sfpu | 930±0%  (0.53x) | 986±1%  (0.66x) | 1067±0%  (0.91x) | 1337±0%  (1.19x) | 1858±0%  (1.53x) | 2902±0%  (1.84x) |

### bf16-fp32

| method | 1t (32e) | 2t (64e) | 4t (128e) | 8t (256e) | 16t (512e) | 32t (1024e) |
|---|---|---|---|---|---|---|
| reduce_fold | 466±1% | 631±0% | 948±0% | 1566±0% | 2816±0% | 5311±0% |
| l1_accum | 810±0%  (0.58x) | 956±0%  (0.66x) | 1253±0%  (0.76x) | 1816±0%  (0.86x) | 2948±0%  (0.96x) | 5150±0%  (1.03x) |
| dest_accum | 846±0%  (0.55x) | 898±0%  (0.70x) | 1030±0%  (0.92x) | 1294±0%  (1.21x) | 1819±0%  (1.55x) | 2868±0%  (1.85x) |
| dest_accum_pairs | 821±0%  (0.57x) | 841±0%  (0.75x) | 889±0%  (1.07x) | 1026±0%  (1.53x) | 1287±0%  (2.19x) | 1806±0%  (2.94x) |
| dest_accum_sfpu | 918±0%  (0.51x) | 945±0%  (0.67x) | 1080±0%  (0.88x) | 1338±0%  (1.17x) | 1872±0%  (1.50x) | 2923±0%  (1.82x) |
| dest_accum_pairs_sfpu | 904±0%  (0.52x) | 913±1%  (0.69x) | 939±0%  (1.01x) | 1073±0%  (1.46x) | 1333±1%  (2.11x) | 1856±0%  (2.86x) |

### bf16-bf16

| method | 1t (32e) | 2t (64e) | 4t (128e) | 8t (256e) | 16t (512e) | 32t (1024e) |
|---|---|---|---|---|---|---|
| reduce_fold | 450±1% | 612±0% | 934±0% | 1557±0% | 2802±0% | 5303±0% |
| l1_accum | 740±0%  (0.61x) | 818±0%  (0.75x) | 1018±0%  (0.92x) | 1347±0%  (1.16x) | 2037±0%  (1.38x) | 3402±0%  (1.56x) |
| dest_accum | 760±0%  (0.59x) | 808±0%  (0.76x) | 942±0%  (0.99x) | 1202±0%  (1.30x) | 1728±0%  (1.62x) | 2774±0%  (1.91x) |
| dest_accum_pairs | 730±0%  (0.62x) | 755±0%  (0.81x) | 801±0%  (1.17x) | 938±0%  (1.66x) | 1194±0%  (2.35x) | 1714±0%  (3.09x) |
| dest_accum_sfpu | 902±0%  (0.50x) | 929±0%  (0.66x) | 1067±0%  (0.88x) | 1328±0%  (1.17x) | 1855±0%  (1.51x) | 2902±0%  (1.83x) |
| dest_accum_pairs_sfpu | 890±0%  (0.51x) | 901±1%  (0.68x) | 921±1%  (1.01x) | 1057±0%  (1.47x) | 1319±0%  (2.12x) | 1839±0%  (2.88x) |

## Accuracy — error vs fp64 mean of the original data  (cell = max_abs \| mean_abs \| max ULP_bf16)

### distribution = signal

**fp32-fp32**

| method | 1t (32e) | 2t (64e) | 4t (128e) | 8t (256e) | 16t (512e) | 32t (1024e) |
|---|---|---|---|---|---|---|
| reduce_fold | 1.3e-03 \| 6.8e-04 \| 0.09u | 1.5e-03 \| 7.2e-04 \| 0.10u | 1.5e-03 \| 7.1e-04 \| 0.10u | 1.5e-03 \| 7.1e-04 \| 0.09u | 1.5e-03 \| 7.1e-04 \| 0.09u | 1.5e-03 \| 7.1e-04 \| 0.10u |
| l1_accum | 1.3e-03 \| 6.8e-04 \| 0.09u | 2.4e-03 \| 1.1e-03 \| 0.15u | 2.6e-03 \| 1.2e-03 \| 0.17u | 3.1e-03 \| 1.3e-03 \| 0.18u | 3.0e-03 \| 1.4e-03 \| 0.21u | 2.7e-03 \| 1.4e-03 \| 0.18u |
| dest_accum | 1.3e-03 \| 6.8e-04 \| 0.09u | 2.4e-03 \| 1.1e-03 \| 0.15u | 2.6e-03 \| 1.2e-03 \| 0.17u | 3.1e-03 \| 1.3e-03 \| 0.18u | 3.0e-03 \| 1.4e-03 \| 0.21u | 2.7e-03 \| 1.4e-03 \| 0.18u |
| dest_accum_pairs | 1.3e-03 \| 6.8e-04 \| 0.09u | 1.7e-03 \| 1.0e-03 \| 0.12u | 2.3e-03 \| 1.2e-03 \| 0.15u | 2.7e-03 \| 1.3e-03 \| 0.15u | 2.6e-03 \| 1.3e-03 \| 0.17u | 2.5e-03 \| 1.3e-03 \| 0.16u |
| dest_accum_sfpu | 1.3e-03 \| 6.8e-04 \| 0.09u | 1.5e-03 \| 7.2e-04 \| 0.10u | 1.5e-03 \| 7.1e-04 \| 0.10u | 1.5e-03 \| 7.1e-04 \| 0.09u | 1.5e-03 \| 7.1e-04 \| 0.09u | 1.5e-03 \| 7.1e-04 \| 0.10u |
| dest_accum_pairs_sfpu | 1.3e-03 \| 6.8e-04 \| 0.09u | 1.1e-03 \| 6.9e-04 \| 0.09u | 1.2e-03 \| 6.8e-04 \| 0.08u | 1.3e-03 \| 6.8e-04 \| 0.08u | 1.2e-03 \| 6.7e-04 \| 0.08u | 1.3e-03 \| 6.8e-04 \| 0.08u |

**bf16-fp32**

| method | 1t (32e) | 2t (64e) | 4t (128e) | 8t (256e) | 16t (512e) | 32t (1024e) |
|---|---|---|---|---|---|---|
| reduce_fold | 1.5e-03 \| 4.8e-04 \| 0.17u | 1.4e-03 \| 3.4e-04 \| 0.10u | 1.9e-03 \| 3.1e-04 \| 0.12u | 7.4e-04 \| 2.0e-04 \| 0.05u | 4.5e-04 \| 1.3e-04 \| 0.03u | 2.6e-04 \| 1.0e-04 \| 0.04u |
| l1_accum | 1.5e-03 \| 4.8e-04 \| 0.17u | 1.4e-03 \| 3.4e-04 \| 0.10u | 1.9e-03 \| 3.0e-04 \| 0.12u | 1.6e-03 \| 2.8e-04 \| 0.06u | 1.1e-03 \| 4.2e-04 \| 0.07u | 1.4e-03 \| 5.3e-04 \| 0.09u |
| dest_accum | 1.5e-03 \| 4.8e-04 \| 0.17u | 1.4e-03 \| 3.4e-04 \| 0.10u | 1.9e-03 \| 3.0e-04 \| 0.12u | 1.6e-03 \| 2.8e-04 \| 0.06u | 1.1e-03 \| 4.2e-04 \| 0.07u | 1.4e-03 \| 5.3e-04 \| 0.09u |
| dest_accum_pairs | 1.5e-03 \| 4.8e-04 \| 0.17u | 1.4e-03 \| 3.4e-04 \| 0.10u | 1.9e-03 \| 3.0e-04 \| 0.12u | 1.6e-03 \| 2.8e-04 \| 0.06u | 1.1e-03 \| 4.2e-04 \| 0.07u | 1.4e-03 \| 5.3e-04 \| 0.09u |
| dest_accum_sfpu | 1.5e-03 \| 4.8e-04 \| 0.17u | 1.4e-03 \| 3.4e-04 \| 0.10u | 1.9e-03 \| 3.1e-04 \| 0.12u | 7.4e-04 \| 2.0e-04 \| 0.05u | 4.5e-04 \| 1.3e-04 \| 0.03u | 2.6e-04 \| 1.0e-04 \| 0.04u |
| dest_accum_pairs_sfpu | 1.5e-03 \| 4.8e-04 \| 0.17u | 1.4e-03 \| 3.4e-04 \| 0.10u | 1.9e-03 \| 3.1e-04 \| 0.12u | 7.4e-04 \| 2.0e-04 \| 0.05u | 4.5e-04 \| 1.3e-04 \| 0.03u | 2.6e-04 \| 1.0e-04 \| 0.04u |

**bf16-bf16**

| method | 1t (32e) | 2t (64e) | 4t (128e) | 8t (256e) | 16t (512e) | 32t (1024e) |
|---|---|---|---|---|---|---|
| reduce_fold | 1.1e-02 \| 3.3e-03 \| 1.36u | 1.1e-02 \| 3.6e-03 \| 0.91u | 1.6e-02 \| 4.4e-03 \| 1.10u | 2.9e-02 \| 8.8e-03 \| 1.84u | 7.1e-02 \| 2.0e-02 \| 4.57u | 2.1e-01 \| 4.8e-02 \| 13.30u |
| l1_accum | 1.5e-03 \| 4.8e-04 \| 0.17u | 1.8e-03 \| 5.9e-04 \| 0.16u | 2.5e-03 \| 6.3e-04 \| 0.22u | 4.5e-03 \| 8.0e-04 \| 0.29u | 4.3e-03 \| 1.0e-03 \| 0.28u | 3.7e-03 \| 1.2e-03 \| 0.24u |
| dest_accum | 1.1e-02 \| 3.3e-03 \| 1.36u | 2.7e-02 \| 4.2e-03 \| 1.70u | 1.4e-02 \| 4.5e-03 \| 0.95u | 1.6e-02 \| 5.6e-03 \| 1.05u | 2.3e-02 \| 7.4e-03 \| 1.95u | 3.3e-02 \| 1.6e-02 \| 2.39u |
| dest_accum_pairs | 1.1e-02 \| 3.3e-03 \| 1.36u | 2.7e-02 \| 4.2e-03 \| 1.70u | 1.4e-02 \| 3.9e-03 \| 0.95u | 2.7e-02 \| 4.1e-03 \| 0.95u | 1.7e-02 \| 5.0e-03 \| 1.39u | 2.1e-02 \| 7.4e-03 \| 1.39u |
| dest_accum_sfpu | 2.8e-02 \| 5.5e-03 \| 0.90u | 1.0e-02 \| 3.3e-03 \| 0.71u | 8.5e-03 \| 2.9e-03 \| 0.62u | 8.2e-03 \| 2.8e-03 \| 0.67u | 1.7e-02 \| 4.6e-03 \| 1.07u | 2.4e-02 \| 1.0e-02 \| 1.68u |
| dest_accum_pairs_sfpu | 2.8e-02 \| 5.5e-03 \| 0.90u | 1.0e-02 \| 3.3e-03 \| 0.71u | 1.2e-02 \| 3.6e-03 \| 0.77u | 1.2e-02 \| 3.4e-03 \| 0.78u | 1.2e-02 \| 3.4e-03 \| 0.77u | 8.2e-03 \| 3.0e-03 \| 0.95u |

### distribution = uniform

**fp32-fp32**

| method | 1t (32e) | 2t (64e) | 4t (128e) | 8t (256e) | 16t (512e) | 32t (1024e) |
|---|---|---|---|---|---|---|
| reduce_fold | 8.9e-05 \| 3.1e-05 \| 0.32u | 5.7e-05 \| 2.2e-05 \| 0.31u | 6.1e-05 \| 1.6e-05 \| 0.36u | 4.3e-05 \| 1.2e-05 \| 3.94u | 2.5e-05 \| 9.0e-06 \| 0.56u | 1.6e-05 \| 5.4e-06 \| 12.52u |
| l1_accum | 8.9e-05 \| 3.1e-05 \| 0.32u | 9.8e-05 \| 3.6e-05 \| 0.33u | 1.1e-04 \| 3.2e-05 \| 2.83u | 5.9e-05 \| 2.3e-05 \| 5.24u | 5.5e-05 \| 1.8e-05 \| 0.64u | 2.6e-05 \| 1.1e-05 \| 9.67u |
| dest_accum | 8.9e-05 \| 3.1e-05 \| 0.32u | 9.8e-05 \| 3.6e-05 \| 0.33u | 1.1e-04 \| 3.2e-05 \| 2.83u | 5.9e-05 \| 2.3e-05 \| 5.24u | 5.5e-05 \| 1.8e-05 \| 0.64u | 2.6e-05 \| 1.1e-05 \| 9.67u |
| dest_accum_pairs | 8.9e-05 \| 3.1e-05 \| 0.32u | 8.1e-05 \| 2.8e-05 \| 0.96u | 9.7e-05 \| 2.5e-05 \| 0.91u | 5.5e-05 \| 2.1e-05 \| 4.11u | 4.9e-05 \| 1.7e-05 \| 1.53u | 2.8e-05 \| 1.0e-05 \| 10.92u |
| dest_accum_sfpu | 8.9e-05 \| 3.1e-05 \| 0.43u | 5.6e-05 \| 2.2e-05 \| 0.22u | 6.0e-05 \| 1.7e-05 \| 0.42u | 4.3e-05 \| 1.2e-05 \| 4.35u | 2.4e-05 \| 9.1e-06 \| 0.66u | 1.7e-05 \| 5.7e-06 \| 12.80u |
| dest_accum_pairs_sfpu | 8.9e-05 \| 3.1e-05 \| 0.43u | 5.9e-05 \| 2.2e-05 \| 0.37u | 5.9e-05 \| 1.6e-05 \| 1.05u | 4.2e-05 \| 1.2e-05 \| 5.11u | 2.6e-05 \| 8.8e-06 \| 0.60u | 1.5e-05 \| 5.5e-06 \| 13.30u |

**bf16-fp32**

| method | 1t (32e) | 2t (64e) | 4t (128e) | 8t (256e) | 16t (512e) | 32t (1024e) |
|---|---|---|---|---|---|---|
| reduce_fold | 3.9e-04 \| 1.3e-04 \| 3.09u | 2.6e-04 \| 1.0e-04 \| 13.08u | 1.8e-04 \| 6.7e-05 \| 8.95u | 1.2e-04 \| 5.1e-05 \| 23.05u | 1.0e-04 \| 3.3e-05 \| 13.28u | 7.2e-05 \| 2.7e-05 \| 38.68u |
| l1_accum | 3.9e-04 \| 1.3e-04 \| 3.09u | 2.5e-04 \| 1.0e-04 \| 13.83u | 1.9e-04 \| 6.7e-05 \| 7.45u | 1.2e-04 \| 5.2e-05 \| 21.83u | 1.2e-04 \| 3.6e-05 \| 15.28u | 7.8e-05 \| 2.7e-05 \| 34.49u |
| dest_accum | 3.9e-04 \| 1.3e-04 \| 3.09u | 2.5e-04 \| 1.0e-04 \| 13.83u | 1.9e-04 \| 6.7e-05 \| 7.45u | 1.2e-04 \| 5.2e-05 \| 21.83u | 1.2e-04 \| 3.6e-05 \| 15.28u | 7.8e-05 \| 2.7e-05 \| 34.49u |
| dest_accum_pairs | 3.9e-04 \| 1.3e-04 \| 3.09u | 2.6e-04 \| 1.0e-04 \| 13.83u | 1.8e-04 \| 6.6e-05 \| 7.45u | 1.2e-04 \| 5.2e-05 \| 22.33u | 1.1e-04 \| 3.6e-05 \| 14.34u | 7.5e-05 \| 2.6e-05 \| 32.99u |
| dest_accum_sfpu | 3.9e-04 \| 1.3e-04 \| 3.09u | 2.5e-04 \| 1.0e-04 \| 13.58u | 1.8e-04 \| 6.7e-05 \| 8.58u | 1.2e-04 \| 5.1e-05 \| 23.02u | 1.0e-04 \| 3.3e-05 \| 13.27u | 7.4e-05 \| 2.7e-05 \| 39.52u |
| dest_accum_pairs_sfpu | 3.9e-04 \| 1.3e-04 \| 3.09u | 2.6e-04 \| 1.0e-04 \| 13.83u | 1.8e-04 \| 6.7e-05 \| 8.95u | 1.2e-04 \| 5.1e-05 \| 23.40u | 1.0e-04 \| 3.3e-05 \| 13.18u | 7.2e-05 \| 2.6e-05 \| 38.08u |

**bf16-bf16**

| method | 1t (32e) | 2t (64e) | 4t (128e) | 8t (256e) | 16t (512e) | 32t (1024e) |
|---|---|---|---|---|---|---|
| reduce_fold | 1.3e-03 \| 2.5e-04 \| 2.49u | 4.4e-04 \| 1.3e-04 \| 3.59u | 7.5e-04 \| 1.8e-04 \| 4.56u | 1.3e-03 \| 2.1e-04 \| 13.39u | 7.4e-04 \| 1.9e-04 \| 31.79u | 7.1e-04 \| 3.0e-04 \| 40.49u |
| l1_accum | 3.9e-04 \| 1.3e-04 \| 3.09u | 5.3e-04 \| 1.5e-04 \| 13.08u | 3.4e-04 \| 1.5e-04 \| 44.95u | 3.5e-04 \| 1.4e-04 \| 39.08u | 3.9e-04 \| 1.2e-04 \| 18.79u | 2.1e-04 \| 9.2e-05 \| 11.61u |
| dest_accum | 1.3e-03 \| 2.5e-04 \| 2.49u | 8.1e-04 \| 2.0e-04 \| 7.99u | 9.6e-04 \| 2.3e-04 \| 16.95u | 7.7e-04 \| 2.1e-04 \| 13.64u | 4.3e-04 \| 1.2e-04 \| 19.79u | 6.3e-04 \| 2.6e-04 \| 328.49u |
| dest_accum_pairs | 1.3e-03 \| 2.5e-04 \| 2.49u | 8.1e-04 \| 2.0e-04 \| 7.99u | 4.8e-04 \| 1.4e-04 \| 17.95u | 3.8e-04 \| 1.4e-04 \| 16.39u | 4.9e-04 \| 1.4e-04 \| 6.68u | 3.7e-04 \| 1.5e-04 \| 111.51u |
| dest_accum_sfpu | 1.3e-03 \| 3.0e-04 \| 3.49u | 1.2e-03 \| 2.0e-04 \| 7.08u | 6.7e-04 \| 1.4e-04 \| 20.95u | 3.6e-04 \| 1.5e-04 \| 13.64u | 3.8e-04 \| 1.1e-04 \| 24.79u | 5.4e-04 \| 2.1e-04 \| 319.49u |
| dest_accum_pairs_sfpu | 1.3e-03 \| 3.0e-04 \| 3.49u | 1.2e-03 \| 2.0e-04 \| 7.08u | 2.7e-04 \| 9.5e-05 \| 17.95u | 3.8e-04 \| 1.1e-04 \| 7.39u | 2.7e-04 \| 8.4e-05 \| 10.68u | 2.8e-04 \| 1.0e-04 \| 131.51u |

### distribution = positive

**fp32-fp32**

| method | 1t (32e) | 2t (64e) | 4t (128e) | 8t (256e) | 16t (512e) | 32t (1024e) |
|---|---|---|---|---|---|---|
| reduce_fold | 2.2e-04 \| 1.6e-04 \| 0.10u | 1.9e-04 \| 1.6e-04 \| 0.09u | 1.9e-04 \| 1.6e-04 \| 0.08u | 1.8e-04 \| 1.6e-04 \| 0.09u | 1.7e-04 \| 1.6e-04 \| 0.08u | 1.7e-04 \| 1.6e-04 \| 0.08u |
| l1_accum | 2.2e-04 \| 1.6e-04 \| 0.10u | 3.2e-04 \| 2.8e-04 \| 0.15u | 4.0e-04 \| 3.3e-04 \| 0.18u | 4.0e-04 \| 3.5e-04 \| 0.19u | 4.0e-04 \| 3.5e-04 \| 0.19u | 3.8e-04 \| 3.4e-04 \| 0.19u |
| dest_accum | 2.2e-04 \| 1.6e-04 \| 0.10u | 3.2e-04 \| 2.8e-04 \| 0.15u | 4.0e-04 \| 3.3e-04 \| 0.18u | 4.0e-04 \| 3.5e-04 \| 0.19u | 4.0e-04 \| 3.5e-04 \| 0.19u | 3.8e-04 \| 3.4e-04 \| 0.19u |
| dest_accum_pairs | 2.2e-04 \| 1.6e-04 \| 0.10u | 2.9e-04 \| 2.1e-04 \| 0.13u | 3.6e-04 \| 2.8e-04 \| 0.16u | 3.8e-04 \| 3.1e-04 \| 0.17u | 3.7e-04 \| 3.2e-04 \| 0.17u | 3.6e-04 \| 3.2e-04 \| 0.18u |
| dest_accum_sfpu | 2.2e-04 \| 1.7e-04 \| 0.10u | 1.9e-04 \| 1.7e-04 \| 0.09u | 1.9e-04 \| 1.6e-04 \| 0.08u | 1.8e-04 \| 1.6e-04 \| 0.09u | 1.8e-04 \| 1.6e-04 \| 0.08u | 1.7e-04 \| 1.6e-04 \| 0.09u |
| dest_accum_pairs_sfpu | 2.2e-04 \| 1.7e-04 \| 0.10u | 1.8e-04 \| 1.4e-04 \| 0.09u | 1.7e-04 \| 1.4e-04 \| 0.07u | 1.6e-04 \| 1.4e-04 \| 0.08u | 1.5e-04 \| 1.4e-04 \| 0.07u | 1.5e-04 \| 1.4e-04 \| 0.08u |

**bf16-fp32**

| method | 1t (32e) | 2t (64e) | 4t (128e) | 8t (256e) | 16t (512e) | 32t (1024e) |
|---|---|---|---|---|---|---|
| reduce_fold | 3.2e-04 \| 1.1e-04 \| 0.17u | 2.8e-04 \| 8.6e-05 \| 0.09u | 2.2e-04 \| 5.1e-05 \| 0.08u | 1.3e-04 \| 4.4e-05 \| 0.05u | 9.3e-05 \| 3.7e-05 \| 0.04u | 5.9e-05 \| 1.6e-05 \| 0.03u |
| l1_accum | 3.2e-04 \| 1.1e-04 \| 0.17u | 2.7e-04 \| 8.2e-05 \| 0.09u | 2.0e-04 \| 8.2e-05 \| 0.10u | 2.5e-04 \| 1.4e-04 \| 0.13u | 3.2e-04 \| 1.7e-04 \| 0.11u | 2.7e-04 \| 1.7e-04 \| 0.10u |
| dest_accum | 3.2e-04 \| 1.1e-04 \| 0.17u | 2.7e-04 \| 8.2e-05 \| 0.09u | 2.0e-04 \| 8.2e-05 \| 0.10u | 2.5e-04 \| 1.4e-04 \| 0.13u | 3.2e-04 \| 1.7e-04 \| 0.11u | 2.7e-04 \| 1.7e-04 \| 0.10u |
| dest_accum_pairs | 3.2e-04 \| 1.1e-04 \| 0.17u | 2.8e-04 \| 8.6e-05 \| 0.09u | 1.8e-04 \| 7.5e-05 \| 0.09u | 2.2e-04 \| 1.3e-04 \| 0.11u | 3.0e-04 \| 1.6e-04 \| 0.11u | 2.7e-04 \| 1.7e-04 \| 0.09u |
| dest_accum_sfpu | 3.2e-04 \| 1.2e-04 \| 0.17u | 2.8e-04 \| 8.5e-05 \| 0.09u | 2.1e-04 \| 5.1e-05 \| 0.08u | 1.3e-04 \| 4.4e-05 \| 0.05u | 9.2e-05 \| 3.6e-05 \| 0.04u | 5.6e-05 \| 1.6e-05 \| 0.03u |
| dest_accum_pairs_sfpu | 3.2e-04 \| 1.2e-04 \| 0.17u | 2.8e-04 \| 8.7e-05 \| 0.09u | 2.1e-04 \| 5.1e-05 \| 0.08u | 1.4e-04 \| 4.4e-05 \| 0.05u | 9.5e-05 \| 3.7e-05 \| 0.04u | 5.9e-05 \| 1.7e-05 \| 0.03u |

**bf16-bf16**

| method | 1t (32e) | 2t (64e) | 4t (128e) | 8t (256e) | 16t (512e) | 32t (1024e) |
|---|---|---|---|---|---|---|
| reduce_fold | 2.3e-03 \| 9.6e-04 \| 0.93u | 2.7e-03 \| 9.5e-04 \| 0.80u | 3.0e-03 \| 1.2e-03 \| 1.03u | 5.6e-03 \| 1.7e-03 \| 1.55u | 7.9e-03 \| 2.4e-03 \| 2.67u | 1.1e-02 \| 4.9e-03 \| 5.83u |
| l1_accum | 3.2e-04 \| 1.1e-04 \| 0.17u | 3.4e-04 \| 1.3e-04 \| 0.11u | 4.5e-04 \| 2.0e-04 \| 0.22u | 6.3e-04 \| 2.1e-04 \| 0.32u | 1.2e-03 \| 3.1e-04 \| 0.29u | 8.6e-04 \| 3.3e-04 \| 0.24u |
| dest_accum | 2.3e-03 \| 9.6e-04 \| 0.93u | 3.0e-03 \| 1.1e-03 \| 1.53u | 4.2e-03 \| 1.3e-03 \| 1.07u | 3.5e-03 \| 1.3e-03 \| 1.77u | 3.8e-03 \| 1.7e-03 \| 1.94u | 5.7e-03 \| 3.5e-03 \| 2.92u |
| dest_accum_pairs | 2.3e-03 \| 9.6e-04 \| 0.93u | 3.0e-03 \| 1.1e-03 \| 1.53u | 2.8e-03 \| 9.5e-04 \| 0.74u | 3.1e-03 \| 1.1e-03 \| 1.30u | 4.0e-03 \| 1.6e-03 \| 1.87u | 4.5e-03 \| 2.0e-03 \| 2.05u |
| dest_accum_sfpu | 3.7e-03 \| 1.3e-03 \| 1.01u | 3.0e-03 \| 1.0e-03 \| 0.80u | 3.0e-03 \| 9.3e-04 \| 0.76u | 2.5e-03 \| 6.4e-04 \| 0.65u | 2.2e-03 \| 8.5e-04 \| 0.94u | 4.0e-03 \| 2.1e-03 \| 2.05u |
| dest_accum_pairs_sfpu | 3.7e-03 \| 1.3e-03 \| 1.01u | 3.0e-03 \| 1.0e-03 \| 0.80u | 3.3e-03 \| 1.1e-03 \| 0.84u | 2.5e-03 \| 6.3e-04 \| 0.65u | 2.6e-03 \| 7.1e-04 \| 0.66u | 2.0e-03 \| 9.4e-04 \| 0.85u |
