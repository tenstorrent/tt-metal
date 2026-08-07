# CLAUDE.md — TT-NN operations

Guidance for coding agents working under `ttnn/cpp/ttnn/operations/`. It covers numerical accuracy
— the failure mode that produces plausible wrong numbers rather than an error — and the
test-writing habits that hide it.

This file loads in addition to the repository-root `CLAUDE.md` and is kept to guidance that applies
only here. See that file for build and debugging guidance, and for what belongs in these files and
how to keep them current; see `.github/copilot-instructions.md` for review standards.

## Numerical accuracy on Tensix

Every operand entering the FPU is truncated to the source register's mantissa. Three consequences
that are easy to get wrong:

- **It is truncation, not round-to-nearest.** The error is one-sided, so it accumulates coherently
  instead of cancelling and does not average out over many elements or over a batch.
- **It applies to data movement, not just arithmetic.** Moving a tile through DEST with `copy_tile`
  truncates it. An op that performs no multiply at all can still lose precision this way.
- **Neither `MathFidelity` nor `fp32_dest_acc_en` recovers it.** The fidelity phases slice a
  mantissa that was already truncated on its way into the source register, and `fp32_dest_acc_en`
  widens DEST, not SrcA/SrcB. Reaching for either as a fix for an accuracy complaint is a common
  wrong turn.

The escape hatch is the SFPU, which carries full fp32. `ReduceFp32Mode`
(`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp:18`) exists for exactly this reason — its own
documentation describes the FPU path as "inputs truncated to tf32 — faster, lossy" and routes fp32
SUM through the SFPU when the caller asks for accuracy. `is_sfpu_reduce_path()` in the same header
gates which ops may take that path. Expect a throughput cost.

Error scales with **the number of elements whose mantissa does not fit** — not with tile count and
not with the number of operations. An op that combines every element of its input (a reduction, a
product) over dense fp32 data can be tens of percent off, while the same op over powers-of-two data
is exact. If an accuracy measurement seems to scale with tiles, check whether the stimulus happens
to place one non-representable value per tile.

Related traps when reasoning about precision in kernels:

- Block-float formats (`bfloat8_b`, `bfloat4_b`) share one exponent across each group of 16 datums,
  so values written into unused lanes of a tile are *not* harmless: they can shift the shared
  exponent and perturb real data. This does not apply to bfloat16.
- On Wormhole and Blackhole, `ELWMUL` accumulates into DEST unconditionally, regardless of
  `dest_accum_en`. A DEST that was not correctly cleared corrupts the result rather than being
  overwritten.

## Writing accuracy tests for an op

- **Compute goldens in float64.** `torch.<op>` on a bfloat16 or float32 tensor performs its own
  reduction in that dtype: it is a second approximation with a different accumulation order, not a
  reference. On a 4096-element product, torch and the device landed 30% apart while each was within
  17% of the exact answer — comparing them measures nothing.
- **Avoid stimuli that are exactly representable** unless that is the point of the test. Powers of
  two, all-ones, and small dyadic values (k/2**n) survive any mantissa width untouched and hide
  truncation completely. They are useful as a deliberate *control* — pairing an exact stimulus with
  a full-mantissa one isolates mantissa width as the cause — but on their own they prove little.
- **Keep the golden finite, and away from denormals.** `torch.allclose(inf, inf)` is `True`, so an
  overflowing golden silently reduces a test to asserting that the device also overflows. At the
  other end, products of many near-1.0 factors drift toward 1e-38 where relative error stops being
  meaningful. Renormalising by an exact power of two moves the result back to O(1) without making
  any element inexact.
- **PCC is skipped for scalar (rank-0) outputs**, so an op returning a scalar needs an explicit
  relative-error assertion; otherwise the tolerance checks are the only thing running.
- **Record where a tolerance came from.** State the architecture it was measured on and the
  stimulus used. A bare `atol=5e-2` cannot later be distinguished from a bound that was widened
  until the test passed, and a bound several times looser than the real error is indistinguishable
  from no coverage at all.
