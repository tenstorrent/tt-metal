# Softmax support for `generalized_moe_gate` — perf findings & decision (DEFERRED)

**Status: deferred.** There is a workaround (do softmax *outside* the gate with `ttnn.softmax`, with
the gate's in-op `sigmoid` turned off — semantically correct: softmax is at the very front of the
gate). Revisit after 512-expert support lands.

## Goal (when we come back to it)
Support softmax routing (over **all** experts) as the gate's scoring function, alongside the current
sigmoid. bias is used **only for top-k selection** (same as the sigmoid version); the output weights
are the renormalized non-bias scores.

## Measured: standalone `ttnn.softmax` (the "outside" option)
Single device, bfloat16, TILE_LAYOUT, L1, shape `[1,1,1,N]` (1 user × N experts), trace-captured
device-kernel time (`tests/.../test_softmax_perf.py` + `perf_softmax.py`):

| experts | µs |
|---|---|
| 64  | 6.8 |
| 128 | 8.1 |
| 256 | 10.6 |
| 512 | 14.87 |

**Linear fit:** ≈ **5.6 µs fixed + 0.018 µs/expert**. So most of the time (esp. at small N) is the
**fixed per-op cost** (program dispatch + kernel launch + L1 read/write), not the softmax compute.

## Conclusion: fusing softmax INTO the gate should be meaningfully faster
- The ~5.6 µs floor is "this is a separate device op" overhead. Fused, softmax piggybacks on the
  gate's already-launched kernel and operates on data already resident in DEST/CBs → that floor is
  removed.
- Also removes the **inter-op data round-trip** and the **layout reformat** (the gate uses an exotic
  16×16-reshape sharded layout; standalone softmax uses standard `[1,1,1,N]` → reformatting between
  them is extra ops/copies).
- Estimate: 256 experts ≈ 10.6 µs external → ~3-5 µs of added exp+reduce compute fused ≈ **~half**;
  bigger savings at 512.

## The work / the catch (when implementing fused)
1. **Full-face Σexp reduction.** softmax over all N experts needs `Z = Σ exp(x)` over the entire
   face — a cross-lane/cross-column reduction in the gate's single-face SFPU layout (the same hard
   class as the top-8 work; rows 0-7 SFPU limit). The gate's existing reductions only go to top-2 /
   top-8; a full Σ is new. `exp` itself is cheap & elementwise (drop-in like `sigmoid_tile`, which
   has no monolithic `softmax_tile` equivalent — softmax = exp + reduce(max) + sub + reduce(sum) +
   recip + mul; see ttnn moreh_softmax).
2. **Z cancels in the OUTPUT.** Final normalize divides selected scores by their sum:
   `exp(x_i)/Z ÷ Σ_sel(exp/Z) = exp(x_i)/Σ_sel(exp)` — Z drops out. So **only top-k SELECTION needs
   the true global Z** (selection ranks by `softmax(x)+bias`, and bias is added *after* the
   nonlinearity so Z does not cancel there). If the selection semantics can tolerate an equivalent
   form (e.g. rank by something that doesn't need global Z), fused softmax is nearly free. Confirm
   against the target golden before committing to the full reduction.

## Bench harness (for re-measuring)
`models/demos/deepseek_v3/tests/test_softmax_perf.py` (trace + signposts, parametrized 64/128/256/512)
and `models/demos/deepseek_v3/tests/perf_softmax.py` (device-perf, logs µs per size; needs a
profiler/tracy build).
