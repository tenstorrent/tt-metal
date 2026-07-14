# Verification Report: rms_norm

Verifier pass over the Phase-0 `rms_norm` implementation (planner → implementer → verifier,
registry model). Op file: `ttnn/ttnn/operations/rms_norm/rms_norm.py`; kernels in `kernels/`.

## Code Review

### Fixes applied (in-code)

1. **`kernels/rms_norm_reader.cpp` — migrated off deprecated NoC API.** `noc_async_read_tile(id,
   TensorAccessor, addr)` is `[[deprecated]]` ("use `noc_async_read_page`"). Replaced with the
   non-deprecated generic `noc_async_read_page(...)` overload (accepts any AddrGen with
   `get_noc_addr()` — the TILE-regime accessor is constructed with `tile_bytes` as page size, so
   the page size resolves correctly). Removes the build warning; behaviour identical.

2. **`kernels/rms_norm_writer.cpp` — same migration.** `noc_async_write_tile` → `noc_async_write_page`.

3. **`eval/golden_tests/rms_norm/axes.py` — `classify_call` was missing `memory_layout`.**
   The op's `validate()` reads `memory_layout` off the tensor and `SUPPORTED` gates on it, but the
   runtime observer omitted it, so captured-only rows (the `test_translated.py` cases, which dispatch
   through the `observed` wrapper) recorded axes with no `memory_layout`. The verifier then read the
   missing axis as an unsupported value and mis-classified **52 passing translated cells as
   `xpass_drift`**. Added `"memory_layout": input_tensor.memory_config().memory_layout` to
   `classify_call`, mirroring the op. This is a golden-harness capture fix (the op file and its
   `SUPPORTED` were already correct); it drove `xpass_drift` 52 → 0 and lifted `supported_pass`
   411 → 463. *(This file is golden-test infra, not `feature_spec.py`; the edit aligns the observer
   with the op, which is the single source of truth the file's own docstring states.)*

### Reviewed and confirmed correct (no change needed)

- **`accumulate_reduce_block()` / `prepare_partial_reduce_scalers()` are stale in `kernel_lib`, and
  the kernels' workaround is the correct idiom.** The compute kernel's header documents that
  `accumulate_reduce_block()` forwards CB ids as *function* args while the current `reduce()` takes
  them as *template* args (`reduce_helpers_compute.hpp:471–487`) — I verified this: the wrapper
  would not compile. The kernel instead calls `reduce<pool, rdim, cb_in, cb_scaler, cb_acc>(...)`
  directly, which is exactly the documented usage (`reduce_helpers_compute.hpp:43–51`). Likewise the
  reader open-codes the full+partial scaler pair with the working pool-type-aware
  `prepare_reduce_scaler<>` instead of the stale `prepare_partial_reduce_scalers()`. Both are
  legitimate helper substitutions, not raw-API bypasses. **Filed as an advisory** (kernel_lib
  staleness is shared-library scope, not this op's) — see Recommendations.
- **Blocking-model / DRY.** `W_BLOCK_TILES` and `ROW_BLOCK_TILES` are single host constants;
  `num_w_blocks = Wt // W_BLOCK_TILES`, all CB page counts, and all loop bounds derive from them.
  **No CB is sized to `Wt`/`W`/sequence length** — the reduce streams `W` in `W_BLOCK_TILES` chunks
  into a bounded 1-tile accumulator, so per-core L1 stays constant for arbitrarily wide `W`
  (empirically: W=32768 runs, no OOM anywhere in the golden suite). No block value is restated as a
  duplicate literal.
- **`cb_sumsq` sized to 2 pages (design table says `row_block_tiles`=1).** This is *necessary*, not a
  deviation bug: the cross-call `Accumulate::at` reload reserves a new accumulator page while the
  previous one is still resident, so a 1-page CB would deadlock. Balanced push/pop confirmed by the
  256-block (W=8192) cases passing with no hang.
- Registry: `void kernel_main()` form, `api/dataflow/dataflow_api.h` include, `TensorAccessor`
  (not `InterleavedAddrGen`), CB push=pop across both regimes, correct broadcast dims
  (`Col` for the per-row `1/rms` scalar, `Row` for the `[1,W]` gamma), no redundant full-tile fills.
- **Entry point does NO host-side layout/format transforms** (`to_layout`/`tilize`/`pad`/…) — the
  prompt's MUST-NOT is satisfied; tilize/untilize live in the compute kernel behind the `rm` regime
  define. Output layout matches input. Validation raises for rank<2 and gamma-dim mismatch (prompt
  MUST). Gamma "none" sentinel always legal (prompt MUST).

### Noted, not fixed (out of scope / deferred)

- **`ROW_BLOCK_TILES` is only half-wired.** It sizes `cb_sumsq` host-side but is not passed to the
  compute kernel, whose outer loop iterates one tile-row at a time (`for t in num_tile_rows`).
  Correct at the Phase-0 value (1); raising it would mis-size `cb_sumsq` without changing compute
  behaviour. Row-blocking is a design-declared future refinement (perf), so this is acceptable now —
  but the knob should be threaded into the compute loop when row-blocking lands (flagged in
  Refinement 3's notes).
- **Reader is latency-bound (read-one-then-barrier).** Both regimes issue one `noc_async_read` +
  one barrier per tile/stick — the anti-pattern `master.md::double_buffer` warns against. This is a
  perf lever, not a correctness issue → Refinement 3 (measured, device-ns gated).
- **Gamma is re-read from DRAM per tile-row per W-block in pass 2** (reuse redundancy the design's
  Blocking-Model lamp calls out). Perf/reuse lever, noted in Recommendations.
- `(void)Wt;` in the compute kernel — passed but unused; harmless, left as-is.

## Registry Conformance

- **Confirmed** `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate()` all present and correctly
  wired. Taggers (`tag_alignment`, `tag_rank`) both have the `(inputs, axes)` signature. Every axis
  the op gates on appears in `SUPPORTED` (dtype, fp32_dest_acc_en, layout, alignment, rank,
  gamma_mode, gamma_dtype, gamma_layout, memory_layout). `validate()` checks SUPPORTED per-axis
  **then** EXCLUSIONS, and the public entry calls `validate()` on its first line.
- **Confirmed the op file does NOT declare `INVALID`** — it is sourced from `feature_spec.py`.
- **No SUPPORTED auto-fix needed** — after the `axes.py` capture fix, `xpass_drift = 0`; the op's
  `SUPPORTED` already matched reality.
- **INVALID audit (`eval/golden_tests/rms_norm/feature_spec.py`)** — well-formed against the three
  sanity rules:
  - Canonical `bf8b + ROW_MAJOR` present for **both** the activation (`{dtype: bf8b, layout: RM}`)
    and the gamma tensor (`{gamma_dtype: bf8b, gamma_layout: RM}`) — each couples axes of a single
    tensor.
  - `no_gamma` ⇄ `"none"` sentinel canonicalization present both ways (present⇒real dtype/layout,
    absent⇒sentinel only) — the canonicalization-only multi-axis exception.
  - No entry couples axes of *different* tensors; no entry encodes a "kernel doesn't support yet"
    (those are EXCLUSIONS). No changes recommended.

## Precision Baseline

Measured via `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py`
(bf16 & fp32, TILE, fp32_dest_acc_en=True, no_gamma rows shown; gamma rows comparable). Ratio =
`actual/expected` over finite non-tiny reference elements — the scale-bug detector.

| Shape | dtype | PCC | Max Abs | Mean Abs | Rel RMS | got/true median (std) | Reading |
|-------|-------|-----|---------|----------|---------|-----------------------|---------|
| (1,1,32,64)   | bf16 | 1.00000 | 0.0156 | 0.00124 | 0.00291 | 1.00000 (0.0028) | clean |
| (2,4,128,512) | bf16 | ~1.0    | 0.0313 | 0.00166 | 0.00348 | 1.00000 (0.0035) | clean |
| (1,1,32,8192) | bf16 | ~1.0    | 0.0625 | 0.00646 | 0.00986 | 0.99320 (0.0064) | precision noise (broad, ~1.0) |
| (1,1,32,4096) | fp32 | ~1.0    | ~0.05  | ~0.008  | ~0.010  | 1.01030 (0.0008) | **scale bias** (tight, off-1.0) |
| (1,1,32,8192) | fp32 | 0.99999 | 0.1030 | 0.01673 | 0.02100 | 1.02087 (0.0011) | **scale bias — xfail (Refinement 1)** |

**Scale-vs-precision triage (probe `probes/probe_011.py`, got/true ratio vs W):**

| W | fp32 ratio (std) | bf16 ratio (std) |
|---|------------------|------------------|
| 512   | 1.00110 (0.0006) | 1.00020 (0.0024) |
| 1024  | 1.00243 (0.0006) | 0.99983 (0.0033) |
| 2048  | 1.00501 (0.0007) | 1.00058 (0.0042) |
| 4096  | 1.01030 (0.0008) | 0.99798 (0.0061) |
| 8192  | 1.02087 (0.0011) | 0.99179 (0.0064) |
| 16384 | 1.04346 (0.0013) | 1.01642 (0.0142) |

**Assessment.** bf16 is genuine precision noise: ratio centred on 1.0, spread widening with W
(accumulation rounding), well inside tolerance. **fp32 is a STRUCTURAL scale bug, not precision**:
the ratio is a *tight* cluster (std ≈ 0.001) at a value that grows **linearly in W** —
`(ratio − 1) ≈ 2.5e-6 · W`. The device mean(x²) is systematically undercounted ∝ W (→ `1/rms`, hence
the output, too large). This is exactly the high-PCC / off-1.0-tight-ratio trap the protocol warns
about: the runner stamps it `severity=precision`, but fp32 intermediates cannot fix it (they're
already fp32). It is routed to a **blocking bug-fix (Refinement 1)**, NOT a precision refinement.
Likely mechanism (for the debugger): the cross-call accumulator reload (`copy_tile` from `cb_sumsq`)
losing fp32 precision per block, or the matmul-with-ones reduce biasing per tile — the design's
anticipated `accumulate + SFPU-finalize` (`AccumulateViaAdd`) reduce datapath is the candidate fix.

**Recommended tolerances (healthy cells):** bf16 PCC ≥ 0.995, rel-RMS ≤ 0.02; fp32 PCC ≥ 0.999,
rel-RMS ≤ 0.02 **for W ≤ 4096** (fp32 W ≥ 8192 fails until Refinement 1).

## Verifier CLI Summary

Results dir: `/tmp/rms_norm_verify2/` (`verifier_report.json`). Golden run: 40425 cases.

- supported_pass: **463**
- xfail_expected: **8018**
- invalid_skipped: **31920**
- no_axes_found: 15 *(all passing `test_regression.py` @numerics tests — call the op directly, not
  via the observer, so they carry no axes; uncharged and benign)*
- **supported_fail: 9** — all `float32 + W=8192` (ranks 2/3/4, gamma & no_gamma). The fp32 scale bug
  above; a **known, tracked** gap (blocking Refinement 1), not a silenced one. float32 stays in
  SUPPORTED (it is correct for W ≤ 4096); there is no shape axis to exclude on and inventing a
  `shape_size` tagger would hide the bug.
- **xpass_drift: 0** ✓ (was 52; fixed via `axes.py`)
- **xfail_wrong_mode: 0** ✓
- supported_marked_xfail: 0 ✓

## Recommendations

- **Refinement priorities & ordering** are in `op_requirements.md`. Refinement 1 (fp32 reduce scale
  bug) is **blocking** — it touches the reduce datapath every later refinement builds on, and per the
  drift protocol no other refinement should land until it is clean.
- **kernel_lib staleness (advisory, out of op scope):** `compute_kernel_lib::accumulate_reduce_block`
  / `accumulate_reduce` (`streaming_reduce_helpers.hpp`) and
  `dataflow_kernel_lib::prepare_partial_reduce_scalers` are stale against the current
  `reduce()` / `prepare_reduce_scaler()` signatures and will not compile. Every future op that wants
  the streaming-reduce wrappers will hit this. Worth fixing in `kernel_lib` (owner: kernel_lib
  maintainers) so ops don't have to open-code the workaround.
- **Gamma reuse (perf, no failing cell):** gamma is re-read from DRAM per tile-row per W-block in
  pass 2. Holding it resident (or mcast-once) is the design's reuse lamp; noted here rather than
  filed because it has no device-ns measurement yet — fold into a perf phase if it profiles hot.
- **L1 pressure:** none observed. CB footprint is constant in `W`; the widest tested case (W=32768,
  bf16) runs. No pre-emptive OOM refinement is warranted.
