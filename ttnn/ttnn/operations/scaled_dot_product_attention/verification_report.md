# Verification Report: scaled_dot_product_attention

Flash-Attention SDPA (tiled online softmax, O(S) memory). Verified at Phase 0
against the registry model. Date: 2026-06-19.

## Code Review

The implementation is clean and idiomatic. No correctness bugs were found; all
helper usage, CB balance, and API conventions check out. No fixes were required.

**What was checked (and passed):**

- **Helper usage.** Every compute phase routes through `kernel_lib` helpers:
  `matmul_block` (QKᵀ with within-tile transpose; PV), `reduce<MAX/SUM,
  REDUCE_ROW>` (row-max / row-sum), `eltwise_chain` (scale, online-softmax
  recurrence m/α/l/O steps, final normalize), `binary_add` (mask), `unary<Recip>`,
  `copy`. The only raw `cb_pop_front` calls release inputs that helpers
  *intentionally* leave fronted (`cb_q` retained by `WaitAndRetainOnLastBlock`;
  `cb_alpha`/`cb_recip` kept by `B=WaitNoPop`; the dead running-max `cb_max`).
  These are the documented caller-owned tail of retain/no-pop policies — correct.
- **CB sync.** Traced push==pop for every CB across the KV recurrence:
  - `cb_q`: reader pushes Dt; QKᵀ retains (num_k=1, no pop); phase L pops Dt. ✓
  - `cb_max`: j=0 reduce pushes 1; each j>0 pops 1 (D2) + pushes 1 (D3); read no-pop by E; drained 1 after loop. Net 1 resident. ✓
  - `cb_sum`: j=0 reduce pushes 1; each j>0 pops+pushes 1 (G); popped by recip. ✓
  - `cb_out_accum`: j=0 matmul pushes Dt; each j>0 pops+pushes Dt (I); popped Dt by final normalize (K). ✓
  - `cb_alpha`/`cb_recip`/`cb_k`/`cb_v`/`cb_mask`/`cb_scores`/`cb_p`/`cb_pv`/`cb_mblock`/`cb_lblock`/`cb_mnew`: push==pop per iteration. ✓
  - `cb_out`: compute pushes Dt; writer waits/pops Dt. ✓
- **TensorAccessor**, not deprecated `InterleavedAddrGen`, for all DRAM↔L1 transfers (Q/K/V/mask/O). ✓
- **Kernel syntax**: `void kernel_main() { }` everywhere; no deprecated namespace pattern. ✓
- **Include paths**: `api/dataflow/dataflow_api.h` / `api/dataflow/noc.h` / `api/dataflow/circular_buffer.h` (not bare). ✓
- **Broadcast dims**: the three per-row broadcasts (E: `scores − m`; I: `α·O`; K: `O·recip`) all use `BroadcastDim::Col` with the stat operand pinned at tile 0 (`CbIndexMode::FirstTile`) — matches the REDUCE_ROW `(N,1)` output shape. Mask add is element-wise (`BroadcastDim::None`). ✓
- **Reduce direction**: both reductions are over the key axis (`W`) → `REDUCE_ROW`, with the two mandatory pool-type-aware scaler CBs (MAX row-0 fill, SUM col-0 fill). ✓

**Design deviations (all documented in-kernel, justified, and validated by the
production SDPA reference — not bugs):**

1. **bf16 intermediate CBs instead of fp32** (op_design.md §Dataflow specifies
   fp32 accumulators). fp32 *DEST* accumulation is retained (`fp32_dest_acc_en=True`);
   only the CB *storage* is bf16. Forced by Issue #13364: a post-matmul / post-reduce
   `pack_reconfig_data_format` to an fp32 pack format hits a `TTI_STALLWAIT(PACK|THCON)`
   that never drains → device hang. Production `sdpa_program_factory.cpp` makes the
   same call ("need to disable fp32 cbs"). This is the root of the precision
   limitation discussed below.
2. **Scale folded after QK (`cb_scores *= scale`) instead of into Q.** A datacopy
   chain immediately preceding `matmul_block` hangs (unpack cannot transition
   datacopy→matmul-AB). Math identical: `scale·(Q·Kᵀ)`; mask still added in scaled
   space. Documented in the kernel header.
3. **`mm_init` boot instead of `compute_kernel_hw_startup` + `mm_block_init`.**
   `mm_init` is a superset; calling both double-inits pack-sync/DEST. Production
   SDPA boots the same way for this matmul+reduce+eltwise mix.

**Minor advisory (not fixed — negligible impact, working kernel):** the reader
reconstructs the mask `TensorAccessor` / `CircularBuffer` handle inside the
per-KV-block loop (`reader.cpp` phase mask). The construction is a lightweight
struct wrap dwarfed by the NoC reads it issues; the cleanest hoist (function-scope
construction) is unsafe when `use_mask==0` because `get_local_cb_interface(cb_mask)`
would read an uninitialized CB. Left as-is to avoid risk in a verified kernel.

## Registry Conformance

- **Confirmed present and correctly wired** in `scaled_dot_product_attention.py`:
  - `INPUT_TAGGERS` — 3 taggers (`tag_alignment`, `tag_attention_kind`,
    `tag_kv_heads`), all with the `(inputs, axes)` signature.
  - `SUPPORTED` — all 7 axes present: `dtype`, `layout`, `alignment`,
    `attention_kind`, `kv_heads_mode`, `mask_mode`, `scale_mode`. Every axis the
    kernel gates on plus every `INPUT_TAGGERS` key appears.
  - `EXCLUSIONS = []`.
  - `validate()` — order is correct: structural shape contract (ValueError) →
    per-axis SUPPORTED (`UnsupportedAxisValue`) → EXCLUSIONS (`ExcludedCell`).
    Called as the first line of the public entry point.
- **Confirmed the op file does NOT declare `INVALID`** — it is a feature_spec.py /
  test-harness concept. ✓
- **No auto-fixes applied to SUPPORTED.** `xpass_drift = 0` — no under-claim.
- **`{mask_mode: causal, attention_kind: cross}` is correctly NOT excluded.** The
  kernel adds whatever dense additive mask block the reader fetched and never
  assumes `S_q == S_kv`. All 10 cross+causal supported cells passed.

### INVALID audit (in `eval/golden_tests/scaled_dot_product_attention/feature_spec.py`)

`INVALID = []` — correct and well-formed:

- **Canonical bf8b+ROW_MAJOR is vacuous.** `TARGET["layout"] = [TILE_LAYOUT]` only
  (SDPA is TILE-only by design — there is no ROW_MAJOR path in the eventual
  universe), so no ROW_MAJOR cell exists in the cartesian product to forbid.
- **No cross-tensor-axis coupling** — there are no INVALID entries, so no authoring
  mistake.
- **No norm-like weight axes** (SDPA has no gamma/beta), so no no-weight
  canonicalization cells are needed.
- **bf8b + non_tile_aligned is not structurally impossible** — bf8b tensors pad to
  tiles on conversion; the partial-tile / block-exponent interaction is a
  precision concern that belongs in `EXCLUSIONS` (handled by the dtype refinement),
  not INVALID.

No change to `feature_spec.py` requested.

## Precision Baseline

Measured by `test_scaled_dot_product_attention_precision_baseline.py` (bf16, TILE,
tile-aligned, MHA self-attention, no mask, auto scale, `torch.randn` inputs,
seed 0). PCC via `comp_pcc`; relative RMS = RMS(err) / RMS(reference).

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-----|-------------|--------------|------------------|
| (1, 1, 32, 32) | 0.99997 | 0.01172 | 0.00163 | 0.01008 |
| (1, 1, 128, 64) | 0.99997 | 0.00977 | 0.00096 | 0.01058 |
| (1, 4, 256, 64) | 0.99997 | 0.01367 | 0.00081 | 0.01113 |
| (1, 8, 512, 64) | 0.99996 | 0.00781 | 0.00059 | 0.01143 |

**Assessment.** On normally-distributed inputs (the registry-matrix distribution),
precision is excellent: PCC ≥ 0.99996 and relative RMS ~1% across all shapes — far
inside the bf16 golden tolerance (PCC 0.995 / RMS 0.05). Relative RMS climbs only
gently with sequence length / head count (1.01% → 1.14%), consistent with bf16
rounding compounding mildly across the online-softmax KV recurrence.

**Distribution sensitivity (from `test_regression.py`).** Adversarial
distributions that drive the softmax toward *uniform* attention over many keys
expose the bf16-CB accumulator limitation more sharply (these are regression
tests, **not** registry cells — all on the same supported `bf16/mha/tile_aligned`
cell):

| Distribution | Worst case | PCC | Rel RMS | Severity |
|---|---|---|---|---|
| large-magnitude (×10) | B1_H12_S512 | 0.9984 | 0.057 | precision |
| uniform [0,1] | B1_H12_S512 | 0.9788 | 0.302 | precision |
| negative-only | B1_H12_S512 | 0.9177 | 0.666 | bug |

The error grows with sequence length (more KV blocks → more recurrence steps →
more bf16 round-trips of the running `l` / `O`). This is the direct consequence of
design-deviation #1 (bf16 CB storage forced by Issue #13364). See Recommendations.

**Recommended tolerances** (unchanged from the shared golden suite — appropriate):
PCC ≥ 0.995, relative RMS ≤ 0.05 for bf16.

## Verifier CLI Summary

From `eval/results/scaled_dot_product_attention/verifier_report.json`
(784 registry rows; the 40 `no_axes_found` are the non-registry `test_regression.py`
numerics tests + the empty `test_op_loose` placeholder — not registry cells):

- supported_pass: **140**
- xfail_expected: **604**
- invalid_skipped: 0 (INVALID is empty)
- supported_fail: **0**   ✓ (ship gate)
- xpass_drift: **0**      ✓ (ship gate)
- xfail_wrong_mode: **0** ✓ (ship gate)
- supported_marked_xfail: 0

All three loud categories are clean. SUPPORTED honestly describes the kernel.

**xfail_expected breakdown** (offending axis values vs SUPPORTED → drives the
refinement queue; every gap is covered by a refinement in `op_requirements.md`):

| Missing (axis = value) | Cells | Refinement |
|---|---|---|
| dtype = float32 | 248 | R1 |
| dtype = bfloat8_b | 248 | R1 |
| kv_heads_mode = gqa | 132 | R2 |
| kv_heads_mode = mqa | 96 | R2 |
| alignment = w_non_aligned | 60 | R3 |
| alignment = h_non_aligned | 60 | R3 |

(Counts overlap where a cell is out-of-SUPPORTED on multiple axes, e.g.
`float32 + gqa`; each such cell is unblocked only when *all* its offending axes
are supported.)

## Recommendations

1. **Refinement ordering** (see `op_requirements.md`): numerical config (R1) →
   GQA/MQA (R2) → non-tile-aligned (R3). R1 is foundational (descriptor-level CB
   format derivation); R3 is last (edge-masking is the trickiest and interacts
   with both the dtype set and the KV-reduction).
2. **GQA/MQA is nearly free in the kernel** — the reader already computes
   `head_group = H/H_kv` and `h_kv = h/head_group`, and the work split is already
   multi-core over `B·H·Sq_t`. R2 is largely "remove the SUPPORTED gate + verify".
   No new multi-core distribution is needed (the op is already grid-parallel), so
   no `/interleaved-parallel` work is implied by any refinement.
3. **bf16-CB precision limitation (no in-scope lever — verification report, not a
   refinement).** The adversarial-distribution regression misses trace directly to
   bf16 CB storage of the running softmax statistics (`cb_max`, `cb_sum`) and the
   running output (`cb_out_accum`), forced by Issue #13364. The obvious lever —
   fp32 CBs — currently *hangs* this LLK (matmul/reduce output `pack_reconfig` to
   fp32). `math_fidelity` is already the bf16-appropriate HiFi2 and `fp32_dest_acc_en`
   is already on, so neither moves the needle on the inter-iteration round-trip.
   This is **not** filed as a refinement (no concrete in-scope lever). Revisit when
   Issue #13364 is resolved, or via a future algorithmic change (e.g. selectively
   fp32 only the pure-eltwise `cb_alpha`/`cb_recip`, which are not matmul/reduce
   outputs — though `cb_max`/`cb_sum` as reduce outputs remain blocked). The bf16
   golden tolerance (PCC 0.995) holds for the registry matrix's `randn` inputs;
   real attention workloads are closer to `randn` than to uniform/negative.
4. **Output dtype on the float32 refinement.** The program descriptor currently
   hard-codes the output tensor to `ttnn.bfloat16`. When R1 adds float32 input
   support, the output dtype must follow the input (or the requested
   `memory_config`/dtype). Flagged for the R1 implementer.
5. **bf8b mask sharp edge (R1).** The golden runner builds the mask in the same
   dtype as Q/K/V. A bf8b additive causal mask must still represent `−inf`
   (large-negative) per block-exponent; if the bf8b mask path misbehaves, the
   `{dtype: bfloat8_b, mask_mode: causal}` cells go to `EXCLUSIONS` per the
   `/numeric-formats-metal` playbook — not a separate refinement.
