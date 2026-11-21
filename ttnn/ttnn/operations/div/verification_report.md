# Verification Report: div

Target: Quasar (`ttnn.ProgramSpec`), backend: ttsim (untimed). Specification: `eval/specs/div/spec.py`
(sha256 `ee6ee1bfe821…`). Reviewed at the implementer's commit `8e1d40c26af`.

## Code Review

Fixed in place (host side, `div.py`; kernels unchanged):

1. **An `int` divisor past the float range raised `OverflowError`.** `div(a, 10**400)` failed in `float(b)` on
   the host, contradicting the design's mechanism cap ("saturate to ±inf, not an `OverflowError`") and R8 (an
   `int` behaves like the float of the same value: float(±10**400) is ±inf). Added `_number_value()`, which
   saturates to ±inf; a / ±inf = 0 on the device. Test: `test_div_thread_plan.py::test_number_kinds`.
2. **A numpy scalar divisor was not a number.** `isinstance(b, (int, float))` rejects `np.float32(2)` (it then
   fell through to the tensor path and failed on `.dtype`). `_is_number` now uses `numbers.Real` (int, float,
   bool, numpy real scalars). Same test.
3. **Default-config resolution was restated in `validate()` and `div()`** (R3: one place). Both now go through
   `_resolve_config()`, which calls `default_compute_kernel_config()` — the only place the default lives.
   `memory_axes(a)` is also computed once (it was computed twice in `validate()`).

Checked and found conformant (no change):

- Kernels: `void kernel_main()`, `api/dataflow/dataflow_api.h` include paths, `TensorAccessor` (no
  `InterleavedAddrGen`), STRIDED rings on implicit sync with one tagged NoC call per entry, and the tag handed
  back as each DM kernel's last statement (`noc_async_{read,write}_set_trid(0, …)`), as the target page requires.
- Ring sync: implicit sync on all three rings; the chain's Upfront wait / AtEnd pop on the input rings and
  Upfront reserve / AtEnd push on the output ring match per NEO (`my_tiles = n/C + (t < n%C)` on both sides
  of the dealing).
- Helper usage: one `eltwise_chain` per block, `CopyTile`/`FillBitcast` → `DivBinary` → `Floor`/`Trunc` →
  `PackTile`; divide and rounding fused in DEST (no intermediate, no second program — GUIDANCE "Rounding
  modes"). `DivUnary` is unavailable on Quasar (static_assert Mul only), so the number is filled into D1 and
  divided by the same `div_binary_tile` as a tensor (R7). No multicast/semaphore code to replace.
- Broadcast efficiency: the number is an RT argument (4 bytes), not a tensor (`number_as_tensor` rejected).
- One dispatch per call: `div()` issues exactly one `ttnn.generic_op`; validation, output allocation and
  argument derivation are host-only.

Deferred to refinements (architectural, not fixable in this pass): none for Phase 0. Advisories below.

## Specification rules

| Rule | Status |
|------|--------|
| R1, R2 | explicit refusals in `validate()` independent of SUPPORTED (hold in every phase); golden: 1680 cells refused as `NotImplementedError` |
| R3 | `default_compute_kernel_config()` exported; `None` resolved only through it (`_resolve_config`); lint passes |
| R4 | `validate()` reads neither `math_fidelity` nor `math_approx_mode`; both are passed to the compute `ComputeHardwareConfig` |
| R5, R6 | floor/trunc cells all pass; precision baseline: 0 mismatches on 65,536 elements per mode/output dtype |
| R7 | number → float32 bits → `FillBitcast` into D1 → the same `DivBinary` as a tensor |
| R8 | `div(a, 2)` == `div(a, 2.0)` (`test_int_divisor_equals_float`); now also numpy scalars and out-of-range ints |
| R9 | `first_operand` not yet in SUPPORTED → refused; Refinement 1 |
| R10 | output dtype from `dtype` or a's; one rounding at pack (bf16 output 0 ulp everywhere measured) |
| R11, R13 | DRAM and L1 cells pass; output in a's buffer, interleaved (`test_output_in_first_operand_buffer`) |
| R12 | every out-of-phase cell refused with `NotImplementedError` (4608 `xfail_expected`, 0 `xfail_wrong_mode`) |
| R14 | `memory_axes` of **each** tensor operand gated against SUPPORTED |
| X1, X2 | `ValueError`; both rejection tests pass |

Structure rules: the specification has no rule with a `structure` part. GUIDANCE items are all followed
(config passed through to the compute hardware config, number as RT arg, fused rounding, accessor reads for
L1, output allocated from a's memory config). Advisory: `ComputeConfigDescriptor` fields other than fidelity,
`fp32_dest_acc_en` and `math_approx_mode` (e.g. `dst_full_sync_en`) are not mapped onto the
`ComputeHardwareConfig`; nothing in the specification asks for them.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from `eval.specs.div.spec`; no op-local axis function.
- `SUPPORTED` lists every axis of TARGET and equals `PHASE0` (lint: PASSED, 0 errors).
- `EXCLUSIONS = []`; `validate()` checks X2, X1, R1/R2, operand dtype/layout agreement, per-operand placement,
  SUPPORTED per axis, then EXCLUSIONS, raising `UnsupportedAxisValue` / `ExcludedCell`.
- `div()` calls `validate()` first. The op file declares no `INVALID`.
- No XPASS drift; no SUPPORTED auto-fix was needed.
- INVALID audit (specification): `INVALID = []`. TARGET has one dtype axis for both operands, so no
  cross-tensor coupling can arise; bfloat8_b is not in TARGET (no Quasar path, target page), so the canonical
  bf8b+ROW_MAJOR entry does not apply; not a norm-like op. R1/R2 combinations are correctly rules (legal but
  refused), not INVALID. No issue.
- `python3 -m eval.validate_contract div --phase0`: PASSED, 0 errors, 6 unresolved items (all the
  specification's own `UNRESOLVED` markers and the two dtype pairs graded by `legacy_default`; none is a defect).

## Design Conformance

- Algorithm, pipeline topology, RISC ownership: as designed (reader → `in_a`/`in_b` → chain → `out` → writer).
- Work distribution: whole rounds of C tiles over the full `compute_with_storage_grid_size()` grid,
  `row_wise=True`; e.g. `(1,2,33600)` engages 32 clusters × 4 NEOs.
- **Thread plan** (read off the host program): `THREAD_PLAN = (4, 4, 2)`, one named constant;
  `num_threads` R/C/W from it on the three `KernelSpec`s; counts passed as CT args; rings sized
  `BUFFER_DEPTH·BLOCK_TILES·C` (per compute thread); all rings implicit sync on both DM kernels, tags handed back.
  The design states the bound as compute (SFPU reciprocal + Newton steps per element), so a C > 1 plan is
  required and is present. Tests present and passing: `test_each_neo_computes_its_own_tiles` (every NEO's
  tiles tagged), `test_blocks_and_plans` (plans (1,1,1)…(4,4,2) × blocks 1/3/4/8, ragged),
  `test_programs_share_a_device_session` (div after other programs incl. a different plan and `to_layout`),
  `test_a_tt_metal_program_follows`. Plan-vs-alternatives measurement: not possible on this backend (ttsim
  has no clock; no `pipeline_logs/model_cycles.jsonl` exists for this phase).
- Blocking-model fidelity: `BLOCK_TILES`, `BUFFER_DEPTH`, `THREAD_PLAN` are each defined once in
  `div_program_artifacts.py` and every dependent quantity (ring entries, CT args, loop trip counts, the ring
  cap check) derives from them; tests turn each knob. No ring scales with a tensor dimension.
- Expression: compute is block-expressed (one chain call per `block_tiles` with upfront wait/reserve and
  end-of-block pop/push). The DM kernels issue one tagged transfer per tile with no per-tile barrier,
  push or wait — the implicit-sync protocol the target page mandates; no per-unit completion boundary exists in
  the kernel, so this is not an expression failure.
- Axis accounting: the ledger states `flat_tile: spans → block_tiles`, the folded axes as streamed; the ring
  capacity scales with `block_tiles` — consistent in both directions.

## L1 Ledger Audit

- Currency: three rings `in_a`, `in_b` (tensor regime only), `out`, each `BUFFER_DEPTH * BLOCK_TILES * C`
  entries of the tensor's page size — matches `create_program_artifacts` exactly. No other buffers.
- Capacity vs live set: capacity = live set (one block per NEO computing + one landing/draining, the
  double-buffer mechanism). No over- or under-sized ring.
- Page format vs DEST width: input pages are the bf16 tensors' (input ABI); the output page is the requested
  output dtype (output ABI, R10), not an intermediate phase boundary — the "16-bit page under 32-bit DEST"
  check does not apply. No float32 page under 16-bit DEST (that DEST width is not supported yet).
- Disjoint lifetimes: none — all three rings are live for the whole program (pipelined), each justified.
- Bounds / closed form: `L1_div = BUFFER_DEPTH·block_tiles·C·(page_a + page_b + page_out)`, closed in
  bounded symbols; 192 KiB (bf16 out) / 256 KiB (fp32 out) on Quasar at Phase 0 knobs, ≤ 6.25 % of 4 MiB.
  The binding cap is the 1 MiB-per-ring DFB limit (host-asserted), which bounds `block_tiles ≤ 32`.
- Data-movement budget: present and matches the code (a, b, out each cross once; number 0 bytes; no
  cross-core traffic). The cheapest-traffic split is the implemented one. Deferred regimes carry positive
  reasons (broadcast: outside PHASE0; `bcast_b_mcast`: disjoint readers after `bcast_b_resident`).
- Block-size defaults: interleaved → full grid, then the block. `BLOCK_TILES = 4` (not the coarsest that
  fits, ≤ 32): justified by per-NEO shares of 1–9 tiles over INPUTS, where a whole-share block would serialize
  read → compute → write; this is a reasoned, not measured, departure (no clock here) and stays a perf lamp.
- Findings: none needing a fix or a fold-in.

## Precision Baseline

`tests/ttnn/unit_tests/operations/div/test_div_precision_baseline.py` (bfloat16 operands, TILE, DRAM, default
config; reference float64 on the stored operands). ULP is the distance in the output dtype; ratio = got/true
over finite non-zero references.

| Shape (a / b) | Out | PCC | Max Abs Err | Mean Abs Err | Rel RMS | ULP max (mean) | ratio p5 / median / p95 |
|---|---|---|---|---|---|---|---|
| (1,1,32,64) / same | bf16 | 0.9999987 | 3.84 | 1.0e-2 | 1.7e-3 | 0 (0) | 0.99733 / 1.00000 / 1.00258 |
| (1,1,32,64) / same | fp32 | 1.0000000 | 1.2e-4 | 3.0e-7 | 5.5e-8 | 2 (0.44) | 0.9999999 / 1 / 1.0000000 |
| (1,2,8400) / 3.0 | bf16 | 0.9999988 | 2.6e-3 | 3.3e-4 | 1.6e-3 | 0 (0) | 0.99764 / 1.00000 / 1.00234 |
| (1,2,8400) / 3.0 | fp32 | 1.0000000 | 1.2e-7 | 7.6e-9 | 3.7e-8 | 1 (0.33) | 1 / 1 / 1.0000001 |
| (4,8,32,256) / same | bf16 | 0.9999988 | 371 | 1.3e-2 | 1.7e-3 | 0 (0) | 0.99741 / 1.00000 / 1.00260 |
| (4,8,32,256) / same | fp32 | 1.0000000 | 1.6e-2 | 3.8e-7 | 5.6e-8 | 2 (0.44) | 0.9999999 / 1 / 1.0000000 |
| (2,512,1024) / same | bf16 | 0.9999999 | 6.8e3 | 1.9e-2 | 1.7e-3 | 0 (0) | 0.99741 / 1.00000 / 1.00260 |
| (2,512,1024) / same | fp32 | 1.0000000 | 0.5 | 7.2e-7 | 1.0e-7 | 2 (0.44) | 0.9999999 / 1 / 1.0000000 |
| (4,8,32,256) floor / trunc | bf16, fp32 | 1.0 | 0 | 0 | 0 | 0 — 0 mismatches | 1 / 1 / 1 |

(The large max-abs values are quotients by divisors near zero — standard-normal b — and are a few bf16 ulp of
very large results; the ULP column shows they are exact.)

**Assessment**: the bfloat16 output is the correctly rounded quotient on every element measured (0 ulp,
the specification's grade). floor/trunc are exact on inputs kept away from integers. The float32 output of
bfloat16 operands is within 2 fp32 ulp (`a · reciprocal₂(b)`, no residual step), far inside the
`legacy_default` (0.999, 0.01) that grades this dtype pair. Ratio median is exactly 1.0 with a symmetric
spread of ±2⁻⁸·⅔ (bf16 rounding) — no scale/structural bug.
**Recommended tolerances**: defaults hold (`NUMERIC_RESOLVED` 0 ulp for bf16/bf16, `legacy_default` for
bf16 → fp32). Forward risk, carried into Refinement 4: float32 operands with a float32 output are graded at
**1 ulp elementwise**, and this kernel's quotient reaches 2 ulp.

## Verifier CLI Summary

Golden suite run twice (before and after the host fixes), identical results; report:
`/tmp/div_golden_v2/verifier_report.json`.

- supported_pass: 435 (432 PHASE0 cells + 3 numeric regression cases `uniform`)
- xfail_expected: 4608 (1680 refused for good by R1/R2; 2928 are TARGET − SUPPORTED, all queued)
- invalid_skipped: 0 (`INVALID = []`)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 3 (the two rejection tests, passed, and the empty `LOOSE_CASES` placeholder, skipped)
- Queue coverage (`--queue op_requirements.md`): 0 gaps over the 8 values of TARGET − SUPPORTED.

Unit tests: `tests/ttnn/unit_tests/operations/div/` 94 / 94 pass (acceptance 55, thread plan / session 27,
precision baseline 12).

## Recommendations

- **Queue order** (op_requirements.md): R1 broadcast (lands the contract of perf R3) → R2 ROW_MAJOR (layout
  tier, carries broadcast) → R3 perf: read a broadcast operand once per cluster (scheme-change; structural
  gate = expanded-operand page reads per call) → R4 float32 + 16-bit DEST (precision tier; extends every path
  once) → R5 perf: SFPU row-scoping on the padded tile-rows of the traced YOLO shapes.
- **Likely bound**: the design reasons the op compute-bound and plans (4, 4, 2).
  The first perf slot targets DRAM
  bytes (broadcast reuse) and the SFPU-scope lever comes after Refinement 4 adds SFPU work, behind a
  `/perf-ceiling-dm` headroom check. Measure the plan against (1,1,1), (1,4,1), (2,4,2) on `ttsim-perf` when
  a timed run exists; on this backend it cannot be measured.
- **Number regime plan**: (4, 4, 2) reads one input with four reader harts; (2, 4, 2) is the design's lamp.
  Not a structural gain, so not queued.
- **Padding work**: the traced YOLO shapes have H = 2, so 30/32 of every tile's SFPU work and DM bytes are
  padding. The DM part is intrinsic to the TILE operand the caller passes; the SFPU part is Refinement 5.
- **Quasar risks for the queue**, each named in its refinement: `UnaryBcast` on Quasar unverified (R1);
  pack-target switch needs `pack_init` with tilize/untilize rings (R2); `ALL`-pattern resident ring unverified
  under C = 4 (R3); `UnpackToDest` inputs with reconfig disabled, and the 1-ulp float32 grade (R4).
- L1 headroom is large (≤ 256 KiB of 4 MiB); no OOM risk on any INPUTS shape at any queued refinement as
  designed. The ring cap (1 MiB) bounds `BLOCK_TILES ≤ 32` at depth 2 with float32 pages.
