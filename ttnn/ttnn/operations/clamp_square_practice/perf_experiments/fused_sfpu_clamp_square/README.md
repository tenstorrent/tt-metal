# perf experiment: fused_sfpu_clamp_square (clamp_square_practice)

Idea: replace the helper chain's two SFPU passes (`Clamp<D0>` then `Square<D0>`) with ONE fused SFPU pass
(load row, max(lo), min(hi), x*x, RNE->bf16, store). Same precision config: bf16, fp32_dest_acc_en=False.

- `kernels/fused_compute.cpp` — graduation candidate (drop-in for the op's compute kernel, same CT/RT args).
  `CS_LOADMACRO=0` (default) plain raw-TTI schedule; `CS_LOADMACRO=1` SFPLOADMACRO schedule.
- `kernels/compute.cpp` — bake-off kernel, `CS_VARIANT` 0..9 (baseline, sfpi variants, raw, stoch, ablations).
- Bench: `tests/ttnn/unit_tests/operations/clamp_square_practice/test_perfexp_fused_sfpu_clamp_square.py`
  (`-k fcs_precision`, `-k fcs_perf`; env filters FCS_VARIANTS / FCS_CASES / FCS_MODES).
  `stub` = reader/writer NoC payload stubbed (compute stage isolated), `full` = the op's real descriptor.

Measured on Blackhole p150b (120 cores), DEVICE KERNEL DURATION, one run per variant:

| variant | stub focus (1,1,4096,4096) | full focus | precision (vs helper / vs torch fp32 ref) |
|---|---|---|---|
| baseline (helper Clamp->Square) | 78.6 us | 181-186 us | — |
| fused_sfpi (sfpi, bounds as locals) | 60.8 us | 175.9 us | bit-identical (bf16-exact bounds) |
| fused_creg (sfpi, bounds in cregs) | 55.9 us | 172.0 us | bit-identical |
| fused_creg_mask (+0xffff0000 mask) | 62.5 us | 177.4 us | bit-identical |
| **raw_rne / final_raw** | **40.5 us** | **164-165 us** | bit-identical |
| **lm_rne / final_lm** (SFPLOADMACRO) | **37.6 us** | **164.5 us** | bit-identical |
| raw_stoch (SFPSTOCHRND nearest-away) | 30.7 us | 162-164 us | 1 ULP on ties (~1-3% of elements) |
| fused_trunc (no rounding) | 42.9 us | 165.2 us | 1 ULP on ~50% (precision probe only) |
| abl_nosfpu (no SFPU at all) | 5.2 us | 154 us | wrong by construction (floor) |

sfpi unroll-8 / paired-row variants hit a compiler ICE (LRA "unable to generate reloads" on min/max vs a
const reg under register pressure) and were dropped; raw TTI replaces them.

Mechanism: SFPSWAP is a 2-cycle instruction on BH, so the clamp costs 4 cycles/row whichever way it is
written; the helper chain additionally pays a second load/store pass, LREG12 reprogramming twice per row and
a 5-op RNE. Fused raw ~11 cycles/row, load-macro ~10 (SFPMUL issued by the macro; stores cannot be macro'd
because the macro delay field is <= 6 instructions).

Non-bf16-exact bounds (-0.3, 0.7): the helper squares the bf16-truncated bound (Clamp stores to 16-bit DEST
first) and is 1 ULP off the op's own fp32 reference on every clamped element; the fused kernels square the
fp32 bound and match the reference exactly.
