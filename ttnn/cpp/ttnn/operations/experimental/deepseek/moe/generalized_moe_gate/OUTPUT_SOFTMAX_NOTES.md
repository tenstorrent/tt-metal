# Output softmax (over the selected top-k) for `generalized_moe_gate` — design & implementation

Status: **✅ DONE** on both the 256 single-op path and the 512 combine path, for k = 4/6/8. Gated by a
new `output_softmax` flag (default false → unchanged linear normalize; deepseek path unaffected). Tests
parametrize `output_softmax ∈ {False, True}` × `topk ∈ {8, 6, 4}`. (Chinese: `OUTPUT_SOFTMAX_NOTES.zh.md`.
This is the **output** softmax over the kept k; the **scoring** softmax over *all* experts at the front
is a separate, still-deferred task — see `SOFTMAX_NOTES.md`.)

## What it computes

- `output_softmax = false` (default): output weights = `s_i / Σ_sel(s_j) * scale` (linear renormalize of
  the selected scores — the existing behavior).
- `output_softmax = true`: output weights = `exp(s_i) / Σ_sel(exp(s_j)) * scale` — **softmax over the
  selected top-k** (Mixtral-style). Selection is unchanged (still ranked by `sigmoid(x)+bias`); only the
  final normalization differs. `scaling_factor` is still applied.

## The key trick — softmax = exp + the existing linear normalize

`linear_normalize(exp(s)) = exp(s_i) / Σ exp(s_j) = softmax(s)`. So we do **not** need a new reduction:
just `exp` the selected scores, then run the **existing** sum → reciprocal → multiply tail. The whole
implementation is one `exp` inserted into `_generalized_moe_gate_finalize_ungrouped`.

## Implementation (`_generalized_moe_gate_finalize_ungrouped<…, topk, output_softmax>`)

```
merge16_core();  store8();              // sorted global top-8 at scores/indices {0,4}
if constexpr (output_softmax) {         // <-- the only new step
    TTI_SETRWC(... SET_D);              // reset Dst RWC so dst_reg base == the TTI/normalize base
    dst_reg[(scores_offset+0)/2] = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(dst_reg[(scores_offset+0)/2]);  // ranks 0-3
    dst_reg[(scores_offset+4)/2] = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(dst_reg[(scores_offset+4)/2]);  // ranks 4-7
}
// top-n mask (zero ranks >= topk)  — unchanged
// normalize tail (Σ scores{0,4} → recip → mul) — unchanged; now sums exp(s) → softmax
```

### Why exp must come BEFORE the top-n mask (critical ordering)

The flow is **exp(all 8) → mask zeros ranks ≥ k → normalize**. If the mask ran first, the dropped ranks
would be `0`, and `exp(0) = 1` — k spurious 1's would pollute `Σexp` and the dropped slots would emit a
nonzero softmax weight. Exp'ing first means the mask zeros the *already-exp'd* values, so dropped ranks
contribute 0 to `Σexp` and output 0. This is why the `if constexpr (output_softmax)` block sits *above*
the `if constexpr (topk …)` mask block. Verified for k=4 (whole offset-4 row zeroed after exp) and k=6
(per-lane mask after exp) — both tested.

### Numerics — no max-subtraction needed

Scores are sigmoid values in `[0,1]` (gate `sigmoid` on, or the input is pre-sigmoided), so `exp ∈ [1,e]`
— no overflow. softmax is shift-invariant, so skipping the usual `exp(x − max)` gives the identical
result while saving a reduce. Uses `_sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>` (21-bit, ~3 FP32 ULP).

### The sparse-lane non-issue (was a worry)

The sorted-8 occupy only 4 lanes per `{0,4}` row (lanes 0,8,16,24; see `TOPN_NOTES.md`); the other 28
lanes hold sort residue. We exp the **whole** 32-lane row, so the residue gets exp'd too. This was a
concern (would `exp(residue)` enter `Σexp`?) — **verified harmless**: the normalize's `SFPTRANSP` + reduce
sums only the 8 valid lanes, the residue is excluded (k=8 softmax matched the torch golden bit-for-tol).

### dst_reg addressing

`dst_reg[k]` ↔ TTI addr `k*SFP_DESTREG_STRIDE` (= `k*2`), default mod-0 (SrcB), which matches the
normalize's mod-0 `TTI_SFPLOAD` of the scores. So `scores+0` (TTI addr 0) = `dst_reg[0]`, `scores+4`
(addr 4) = `dst_reg[2]`. Reset Dst RWC first so the sfpi base lines up with the TTI base (same gotcha as
the top-n mask).

## Plumbing (`output_softmax`, a named compile-time arg)

`op.py` / nanobind (`output_softmax=False`) → `generalized_moe_gate` op entry → `device_operation::invoke`
→ `operation_attributes_t.output_softmax` → `program_descriptor_builder` (`{"moe_gate_softmax", …}`) →
`ComputeCTArgs::output_softmax` → `generalized_moe_gate_kernel.cpp`
(`get_named_compile_time_arg_val("moe_gate_softmax")`) → the gate template / `combine_finalize<is_32bit,
topk, output_softmax>` → `finalize_ungrouped`. `hash_moe_gate_program_structure` hashes
`named_compile_time_args`, so each (topk, output_softmax) combo gets its own compiled program.

Key files: `ckernel_sfpu_generalized_moe_gate_topk_single_face.h` (the exp block + `#include
"ckernel_sfpu_exp.h"`), `compute_kernel_api/generalized_moe_gate.h` (gate template + `combine_finalize`),
`unified_kernels/generalized_moe_gate.hpp` (`CTArgs::output_softmax` passed to both paths), and the usual
op/device/descriptor/nanobind/op.py stack.

## Golden & tests

`op.py` golden: `weights = exp(topk_scores) if output_softmax else topk_scores`, then `weights / Σweights
* scaling`. `test_generalized_moe_gate` (256) and `test_generalized_moe_gate_512_global` (512) both
parametrize `output_softmax`; the 256 test's self-consistency check also branches to the softmax form.

## Not this

Scoring softmax over **all** experts at the front (needs the global `Σexp`) — deferred, see
`SOFTMAX_NOTES.md`.
