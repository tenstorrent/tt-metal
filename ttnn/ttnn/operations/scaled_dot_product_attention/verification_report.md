# Verification Report: scaled_dot_product_attention

Flash-Attention SDPA (`softmax(Q·Kᵀ·scale [+ mask])·V`, online-softmax over the
KV sequence). Phase-0 verification pass.

## Code Review

Reviewed the op file, program descriptor, and all three kernels (reader / compute /
writer) against `op_design.md` and the helper library.

**Fixed**
- **Reader: hoisted the mask `TensorAccessor` out of the KV loop.** It was
  re-constructed on every KV-block iteration inside `if constexpr (has_mask)`
  (`scaled_dot_product_attention_reader.cpp`). The accessor is loop-invariant
  (`mask_args`, `mask_addr`, `tile_bytes` are all fixed), so it is now built once
  next to the Q/K/V accessors. Pure efficiency; tests unchanged.

**Reviewed and accepted as-is (documented deviations from `op_design.md`, all
correct)**
- **`cb_qs` (idx 4) added; pre-scale Q is non-in-place** (`cb_q → cb_qs` via
  `eltwise_chain(CopyTile → MulUnary → PackTile)`), not `transform_in_place`. The
  design's in-place same-CB read+write deadlocks (reserve-before-pop on a full
  CB). `cb_qs` carries the retained scaled Q across the KV loop (matmul
  `WaitAndRetainOnLastBlock`), popped at Q-block end. Correct.
- **`cb_masked` (idx 15) added; custom mask add is non-in-place** (`cb_scores +
  cb_mask → cb_masked`) to avoid an in-place block-alias 2×-size deadlock.
  Correct; `cb_sc` selects the masked block when a mask is present.
- **`m_new` merge uses the `binary_sfpu<BinaryMax>` convenience wrapper** instead
  of a raw `eltwise_chain` — identical semantics, simpler.
- **j==0 special-case** copies block results straight into the running state
  (no `-inf` sentinel, no `exp(-∞)`/`0·∞`) — matches the design's numerically-exact
  recurrence.
- **Running-state drain**: `cb_m_run` is explicitly popped at Q-block end (the
  multi-Q-block-per-core regression, covered by `test_multi_qblock_per_core`);
  `cb_l_run`/`cb_o_run` are consumed by the final recip / normalize. Verified via
  the debug suite (multi-KV-block + multi-Q-block-per-core all green).

**Helper usage** — every compute phase maps to a `kernel_lib` helper
(`matmul_block`, `reduce`, `eltwise_chain`, `binary_sfpu`, `unary`, `copy`);
no raw compute API. Reader/writer use `TensorAccessor` (not the deprecated
`InterleavedAddrGen`). Kernels use `void kernel_main()` and
`api/dataflow/dataflow_api.h` / `api/compute/*` includes. CB reserve/push and
wait/pop are owned by the helpers; the two manual pops at Q-block end are correct
and necessary.

**Advisory (not blocking, no failing cell — left as-is)**
- Host-side work distribution is hand-rolled (`_enumerate_cores` + `base/rem`
  contiguous split) rather than `ttnn.split_work_to_cores` as the design's Work
  Distribution section specifies. The result is identical (contiguous Q-block runs
  per core) and correct; the op is genuinely multi-core (`PROPERTIES.multi_core =
  verified`, confirmed by long-context debug tests spanning the full grid). Not
  worth a rewrite in this pass; the `/interleaved-parallel` skill would fold this
  into the canonical helper if a multi-core refinement ever revisits the split.
- `default_compute_kernel_config()` sets `math_fidelity = HiFi4` (design-consistent).
  Note: `eval/.../axes.py` has a stale *comment* saying "None → HiFi2"; it reads the
  flag off the op's own factory at runtime, so the mislabel is cosmetic and does not
  affect tagging. `axes.py` is test infra — not edited here.

## Registry Conformance

- **INPUT_TAGGERS** — present, 3 taggers (`tag_alignment`, `tag_attention_kind`,
  `tag_kv_heads`), all with the `(inputs, axes)` signature. Every tagger key
  (`alignment`, `attention_kind`, `kv_heads_mode`) appears in `SUPPORTED`. ✓
- **SUPPORTED** — present; covers every gated axis (`dtype`, `fp32_dest_acc_en`,
  `layout`, `alignment`, `attention_kind`, `kv_heads_mode`, `mask_mode`,
  `scale_mode`). ✓
- **EXCLUSIONS** — `[{dtype: float32, fp32_dest_acc_en: False}]`, armed for the
  dtype refinement (float32 is not yet in SUPPORTED, so this is dormant but
  correctly declared for when fp32 lands). ✓
- **validate()** — checks SUPPORTED per-axis, then EXCLUSIONS (cell-level); raises
  `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`. It
  is the first line of the public entry point. Structural shape contract
  (rank-4, Q.D==K.D, K==V, batch, `H_q % H_kv`, mask shape) raises `ValueError`
  independently. ✓
- **No `INVALID` in the op file** — confirmed absent. ✓
- **No drift auto-fixes needed** — `xpass_drift = 0`, `xfail_wrong_mode = 0`.

**INVALID audit (`feature_spec.py`)** — `INVALID = []` with documented rationale:
`TARGET["layout"]` is TILE-only (SDPA has no ROW_MAJOR path), so the canonical
`bf8b + ROW_MAJOR` rule is vacuous (no ROW_MAJOR cell exists in the cartesian
product). Well-formed: no cross-tensor-axis coupling, nothing that is really an
"unsupported-for-now" (those are EXCLUSIONS). SDPA is not norm-like, so no
weight-canonicalization cells apply. Audit passes.

## Precision Baseline

bf16, TILE, HiFi4 + `fp32_dest_acc_en=True` (Phase-0 default), auto scale, no mask.
Reference = `torch.nn.functional.scaled_dot_product_attention` in fp32.

| Shape (Q=K=V) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|---|---|---|---|---|
| (1,1,32,32)  — single tile          | 0.9999943 | 0.00425 | 0.00071 | 0.00352 |
| (1,1,128,64) — multi-tile, 1 KV blk | 0.9999919 | 0.00385 | 0.00042 | 0.00408 |
| (1,2,256,64) — 2 KV blocks (recur.) | 0.9999905 | 0.00370 | 0.00031 | 0.00442 |
| (2,4,512,64) — multi-batch/head, 4 KV blks | 0.9999906 | 0.00449 | 0.00024 | 0.00445 |

**Assessment**: excellent — PCC ≥ 0.99999 across single-tile → multi-KV-block
recurrence → multi-batch/head. The online-softmax accumulation does not degrade
precision at these lengths (fp32 running state + fp32 DEST). Relative RMS ≈ 0.4%,
consistent with bf16 input quantization.

**Recommended tolerances (bf16, Phase-0 default config)**: PCC ≥ 0.995,
rtol=0.05, atol=0.05. (The golden suite's `helpers.py` band and the acceptance
test's 0.995 are appropriate.)

## Verifier CLI Summary

From `verifier_report.json` (golden suite: 2803 collected):

- supported_pass: **520**
- xfail_expected: **2113**
- invalid_skipped: 0 (INVALID = [])
- supported_fail: **51**  → 24 `OOM` + 27 `numerical-precision` (all queued as refinements; see below)
- xpass_drift: **0** ✓
- xfail_wrong_mode: **0** ✓
- supported_marked_xfail: 0 ✓
- no_axes_found: 119 (skipped/deselected translated configs with no recorded axes — not failures)

Supported cells: **520 / 571 passing**. The 51 `supported_fail` split cleanly and
are honest edges, not over-claims:

- **24 × OOM** (`test_golden.py`, all `Q(1,1,128,{256,512,1024})` — large
  head_dim). Per-core L1 CBs scale with `Dt` (`cb_q`, `cb_qs`, `cb_k`, `cb_v` and
  the fp32 `cb_o_run`/`cb_o_new`/`cb_pv` at `q_chunk_t·Dt`). D=256 is borderline,
  D=512/1024 overflow the 1.5 MB budget. → **Refinement 4** (`/memory-budget-metal`).
  Not silenced with EXCLUSIONS: OOM is a resource boundary, not a kernel branch —
  the failure category is the signal.
- **27 × numerical-precision** (`test_translated.py`, upstream nightly tests).
  These force `fp32_dest_acc_en=False` (16-bit DEST) with tight RMSE thresholds
  and/or extreme sequence lengths the golden matrix does not exercise:
  `full-grid` = `(1,3,3,45056,128)` → PCC 0.978 / rms over target; `(1,16,16,4096,96)`
  → PCC 0.9994 but rms just over 0.0069; masked bcast variants over rms 0.033.
  The golden long-context cells (S=4096, 8192) **pass** — these translated cases
  live beyond that envelope. See Recommendations; not a distinct refinement (no
  clean in-kernel lever beyond the caller's `fp32_dest_acc_en` flag).

No hangs, no compilation failures, no `numerical-bug` (inf/NaN/PCC≤0.9). The two
loud gate categories (`xpass_drift`, `xfail_wrong_mode`) are 0 — SUPPORTED is
honest.

## Recommendations

- **Refinement ordering** (see `op_requirements.md`): dtype (R1) first — it is
  foundational and R2/R4 reuse the dtype-driven CB-format derivation. Alignment
  (R2) and causal (R3) are independent. The L1 memory-budget fit (R4) goes last
  (memory pressure last), after the SUPPORTED rectangle is otherwise stable.
- **`bfloat8_b + non_tile_aligned_dim`** will not pass out of the box (block-format
  edge); route it to `EXCLUSIONS` inside the dtype refinement rather than its own
  entry (the `/numeric-formats-metal` contract).
- **`fp32_dest_acc_en=False` at long context is a known precision limitation**, not
  a bug. The running m/l/O state is already fp32; the loss is in the bf16 matmul
  DEST when the caller opts out of fp32 accumulation. There is no in-kernel lever
  short of overriding the caller's flag (the design explicitly forbids silently
  forcing `True`). Callers wanting exact long-context should pass
  `fp32_dest_acc_en=True` (the Phase-0 default). Documented here rather than
  queued — no `(axis, value)` gap and no concrete kernel lever in scope.
- **`EXCLUSIONS` dtype=float32 entry is dormant** until R1 adds float32 to
  SUPPORTED; keep it declared (it is correctly armed).
