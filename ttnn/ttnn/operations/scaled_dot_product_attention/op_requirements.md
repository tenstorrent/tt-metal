# Operation Requirements: scaled_dot_product_attention

## Definition
- **Formula**: `O = softmax(Q·Kᵀ·scale + mask) · V`, computed via FlashAttention-2
  (tiled online-softmax over KV blocks; the `S_q × S_kv` score matrix is never
  materialized). `scale` defaults to `1/sqrt(D)`.
- **PyTorch Reference**:
  ```python
  def sdpa(q, k, v, attn_mask=None, scale=None):
      B, H, Sq, D = q.shape
      Hkv = k.shape[1]
      if scale is None:
          scale = 1.0 / math.sqrt(D)
      if Hkv != H:                                   # GQA / MQA
          rep = H // Hkv
          k = k.repeat_interleave(rep, dim=1)
          v = v.repeat_interleave(rep, dim=1)
      scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
      if attn_mask is not None:
          scores = scores + attn_mask.float()
      return torch.matmul(torch.softmax(scores, dim=-1), v.float())
  ```
- **Import Path**: `from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention`
- **Function Signature**:
  ```python
  scaled_dot_product_attention(
      query: ttnn.Tensor,                          # (B, H, S_q, D)
      key: ttnn.Tensor,                            # (B, H_kv, S_kv, D)
      value: ttnn.Tensor,                          # (B, H_kv, S_kv, D)
      *,
      attn_mask: ttnn.Tensor | None = None,        # (B, {1,H}, S_q, S_kv) additive
      is_causal: bool = False,                     # mutually exclusive with attn_mask
      scale: float | None = None,                  # None → 1/sqrt(D)
      compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None,
      memory_config: ttnn.MemoryConfig | None = None,
  ) -> ttnn.Tensor                                 # (B, H, S_q, D)
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N`. A `[~]` partial's sharper follow-up appends a lowercase letter to the parent's number (`Refinement 1b`, `Refinement 1c`, …), ordered immediately after its parent. The runner's parser matches exactly `Refinement \d+[a-z]?`.

> **TARGET − SUPPORTED gap covered by this queue** (everything else in TARGET is
> already SUPPORTED and green):
> - `dtype`: float32, bfloat8_b → **R2**
> - `fp32_dest_acc_en`: False → **R2**
> - `alignment`: w_non_aligned, h_non_aligned → **R1**
> - `mask_mode`: causal → **R4**
>
> No `memory_layout`/sharding axis exists in the golden TARGET, so flash-decode
> (cross-core `S_kv` split) and GQA KV-multicast are **not** queued — they have no
> golden cell to unlock (see verification_report.md "Observations").

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED alignment**: [tile_aligned]
- **SUPPORTED attention_kind**: [self, cross]
- **SUPPORTED kv_heads_mode**: [mha, gqa, mqa]
- **SUPPORTED mask_mode**: [none, custom]
- **SUPPORTED scale_mode**: [auto, explicit]
- **EXCLUSIONS**: none
- **Cores**: multi-core (`split_work_to_cores` over `B·H·n_q_chunks`, interleaved DRAM)
- **Blocking**: `Sq_chunk_t`/`Skv_chunk_t`/`KV_DEPTH` fitted once by `_fit_l1`
- **Compute config**: HiFi4 + fp32 DEST (default), knobs threaded from `compute_kernel_config`
- **Golden baseline**: **212 / 212** supported cells passing (verifier CLI: supported_fail=0, xpass_drift=0, xfail_wrong_mode=0)

---

### [x] Refinement 1 — Non-tile-aligned shapes (w_non_aligned + h_non_aligned)

**Goal**: add `"w_non_aligned"` and `"h_non_aligned"` to `SUPPORTED["alignment"]`,
handled **natively in the kernel** (no `ttnn.tilize`/`to_layout` wrapper). Covers
both non-aligned legs in one refinement (they share the reader/compute edge
machinery):
- **`w_non_aligned` (D % 32 ≠ 0)**: zero-pad the last D-tile in the reader so the
  Q·Kᵀ contraction (over `Dt`) and the P·V free-dim are correct; drop the padded
  output columns of the last D-tile in the writer.
- **`h_non_aligned` (S_q or S_kv % 32 ≠ 0)**: the last Q-chunk writes only valid
  rows; **the last KV tile's padding rows must be masked to −∞ before the
  row-max/exp/sum** so they fall out of the softmax denominator — use the reduce
  helper's partial-scaler / mask path (`ReducePartialScaler::last_tile_at` +
  `calculate_and_prepare_partial_reduce_scalers` / `prepare_reduce_mask` on the
  reader). This is the structural part — attention's softmax reduction spans the
  padded KV axis, so a plain edge-pad is not enough.

**Implementation skill**: /memory-layouts

**Verifier notes**: **hardest generality refinement — land it first, on the
smallest test surface** (before R2 adds float32/bf8b, so the edge machinery is
validated on bf16 alone). While here, replace the descriptor's `_chunk_size`
largest-divisor trick with a coarse chunk (`min(axis_t, 4)`) + a masked partial
remainder (the divisor trick avoids partial chunks today but would collapse to a
1-tile chunk for a prime tile-count > 4 — see verification_report.md). The skill
body owns the last-tile zero-pad/mask pattern; do not re-derive it here.
`bfloat8_b + non_tile_aligned` is expected to fail and becomes an EXCLUSION armed
by R2, not by this refinement.

**Done when**: the `alignment ∈ {w_non_aligned, h_non_aligned}` golden cells
(currently `xfail_expected`) pass; the mask-reduction correctly excludes KV
padding (verify on `test_regression`-style partial-S_kv shapes); prior phases green.

**Landed (R1)**: both alignment values added to `SUPPORTED`; golden **252 passed**
(212 prior + 40 non-aligned), **0 failed / 0 xpass**; prior unit + regression
unchanged (the 9 `test_regression` misses are the same pre-existing aligned-
adversarial precision cells, R2's target). `w_non_aligned` (D%32) rides the
`from_torch` tile zero-padding through the QKᵀ contraction and PV free dim (no
reader change); `h_non_aligned` S_q writes only valid rows (whole-tile write +
logical slice); the structural piece — S_kv%32 — is an additive **−∞** mask
(bf16 `0xFF80`, face-aware `fill_vertical_mask_tile` in the reader, mirroring
production `fill_vertical_tile_bf16`) added to the last KV chunk's boundary tile
before the row-max/exp/row-sum, reusing the existing additive-mask `add` path.
The `_chunk_size` divisor trick was **kept** (correct + DRY + general for every
tested/realistic shape) and its generalization deferred to R1b below.

### [x] Refinement 1b — Coarse-chunk + partial-remainder (replace `_chunk_size` divisor trick)

**Goal** (verifier-noted "while here" from R1, deferred): replace
`_chunk_size(axis_t, target)`'s largest-divisor rule with a coarse chunk
`min(axis_t, 4)` plus a **partial last chunk** (fewer whole tiles than
`Sq_chunk_t`/`Skv_chunk_t`). Today the divisor trick keeps every chunk whole
(so the *only* partial unit is the last S_kv tile's columns, handled by R1's
mask), but for a **prime tile-count > 4** (e.g. `S_kv = 32·101 → Skv_t=101`) it
collapses to a 1-tile chunk — a granularity-floor violation that repays per-chunk
reconfig/init/fill-drain overhead every tile.

**Exact lever**: thread a per-chunk runtime tile count
`min(chunk_t, axis_t − j·chunk_t)` into the reader's read counts, the compute
`MatmulBlockShape`/`ReduceInputBlockShape`/`EltwiseShape` (all take runtime
extents) with a re-derived matmul subblock decomposition for the partial `n`,
and the writer's tile counts — for **both** the Sq q-chunk and the Skv loop. This
is a core-loop restructuring touching all three kernels (regression risk to the
212 aligned + perf-flagged shape), so it is split out rather than bundled into R1.

**Verifier notes**: **no current test exercises a prime tile-count > 4** (golden
`S_kv` are all composite or chunk-4-divisible; worst real case `Skv_t=6 → 3`), so
this is a generality/perf hardening with no golden cell to unlock — add a
`test_regression`-style shape (e.g. `S_kv=3232`, `Skv_t=101`) alongside it.
Gate on the full golden suite staying green (no regression) as the acceptance net.

**Done when**: `_chunk_size` uses the coarse-chunk + partial-remainder scheme; a
prime-`Skv_t`(>4) shape runs at chunk 4 (not 1) and is correct; the full golden
suite + unit + regression stay green.

**Landed (R1b)**: `_chunk_size` now returns a coarse chunk `min(axis_t, target)`
with a **partial last chunk** of `axis_t % chunk` tiles; the reader/compute/writer
thread a per-chunk runtime tile count `min(chunk, axis_t − j·chunk)` — `sq_valid`
(M extent, per q-chunk work unit; compute decodes `qc` from `start_wu`) and
`skv_valid` (QKᵀ N / PV K, per KV chunk) — through the read counts, the
`MatmulBlockShape`/`ReduceInputBlockShape`/`EltwiseShape` runtime extents, and the
write counts, for **both** axes. The matmul N-subblock decomposition is derived
**on-device** (`decomp_n`, replacing the host `_matmul_subblocks` — single source
of truth). Prime `Skv_t`=101 → **chunk 4** (26 chunks, last=1), not the divisor
trick's chunk 1; no prime collapses to 1. **Constraint discovered on device:** the
partial remainder must **divide** the chunk (`rem | chunk` ⇔ `2·rem ≤ chunk`) — the
score-block CBs (`cb_scores`/`cb_exp`) are ring buffers read by linearly-indexed
compute (row-max reduce + exp), and the in-place mask `add` rotates the read pointer
by the per-chunk tile count; a remainder that doesn't divide the chunk offsets the
reduce window past the ring wrap → OOB unpack → packer wedge (`Skv_t=7` at chunk 4
gave remainder 3, `2·3>4`, and hung). So `_chunk_size` picks the largest straddle-safe
chunk ≤ target (`Skv_t=7 → 3`, `Skv_t=11 → 2`, `Skv_t=101 → 4`, `Skv_t=296 → 4`);
the granularity floor is raised from 1 to ≥2 (and 4 wherever `rem | 4`). QKᵀ
`out_subblock_w` is held **constant** across the KV loop (optimal when no partial
chunk — incl. the perf-flagged `Skv_t=296` shape; `1` for partial-chunk shapes) so
`mm_block_init_short` never reconfigs the packer to a partial width mid-loop.
Golden **252 passed / 2017 xfailed** (0 fail, 0 xpass — no SUPPORTED change); unit
**62/62** (prod mode); `test_regression` unchanged (same 9 R2-target precision
misses). A strict *chunk-4-for-every-prime* variant would need a pad-to-full +
`−∞`-mask scheme (compute always full blocks) — deferred; no test/golden cell needs
it and the perf delta is marginal.

### [x] Refinement 2 — Numerical configurability (dtype + compute-config + intermediate precision)

**Goal**: add `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, add
`False` to `SUPPORTED["fp32_dest_acc_en"]`, expose the caller's
`compute_kernel_config` (`math_fidelity`, `fp32_dest_acc_en`, `math_approx_mode`)
end-to-end, and correct intermediate-CB precision (`cb_scores`/`cb_exp` fp32 where
it pays, `UnpackToDestFp32` tagging). Cells that fail out of the box land in
`EXCLUSIONS`, **not** their own refinement — arm at minimum:
- `{"dtype": ttnn.float32, "fp32_dest_acc_en": False}` — maxed input + non-maxed
  accumulation is lossy; refuse it (honor the caller's flag, don't silently force
  True). The golden `TOLERANCES` omits this combo for the same reason.
- `{"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"}` (and `h_non_aligned`)
  if bf8b + non-aligned misses tolerance — the canonical `/numeric-formats-metal`
  EXCLUSION. (Depends on R1 having added the non-aligned values first.)

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: **pulled ahead of the (harder) causal refinement (R4)** because
the perf-flagged loose case `1×10×9472×128` pins `fp32_dest_acc_en=False` — that
shape is `xfail` until this refinement lands it in SUPPORTED, and **R3 (perf)
cannot run against it until then**. Also fixes/relaxes the 9 `test_regression.py`
precision failures (float32 dtype + fp32 intermediates push the adversarial
distributions back under tolerance — see verification_report.md; they are genuine
bf16 precision, ruled not-a-bug). Order after R1 so the bf8b + non-aligned
EXCLUSIONS have their axis values to reference. Pass condition per the skill: zero
kernel changes when helpers/descriptor precision are wired correctly.

**Landed (R2)**: `SUPPORTED["dtype"] += {float32, bfloat8_b}` and
`SUPPORTED["fp32_dest_acc_en"] += False`, with **zero compute-kernel changes** (the
pass condition held — every phase is helper-based, no hard-coded formats). All
descriptor-level: `cb_scores`/`cb_exp` promoted to **Float32 under fp32-DEST**
accumulation (bf16 otherwise, so the perf-flagged bf16+16-bit-DEST shape stays
byte-identical); consumed by FPU ops (add/reduce/sub/matmul) so **not**
`UnpackToDestFp32`-tagged — the L1 format alone lifts bf16→TF32 through the softmax.
`_fit_l1`/`_working_set_bytes` now use real per-dtype tile bytes + the intermediate
format (fp32 input no longer under-counts L1). `compute_kernel_config` threaded
end-to-end (added `dst_full_sync_en`; math_fidelity/fp32_dest_acc_en/math_approx_mode
were already wired). **EXCLUSIONS armed** (all verifier-pre-authorized): `{float32,
fp32_dest_acc_en=False}` (maxed input + non-maxed acc, honored not forced) and
`{bfloat8_b, w_non_aligned}` + `{bfloat8_b, h_non_aligned}` — the canonical
block-float × partial-tile incompatibility (measured PCC ≈ 0.2–0.5 on the `S_kv%32≠0`
additive-−∞ mask path; both alignment tags carry it, so both refused; bf8b +
tile_aligned is fully supported). Golden **1181 passed / 1088 xfailed** (0 failed,
0 xpass — no drift), up from 252 at R1b; the perf-flagged loose case
(`1×10×9472×128`, bf16, `fp32_dest_acc_en=False`) now **runs and passes**, unblocking
R3. Unit **62/62**; translated bf16 sanity green. New
`test_scaled_dot_product_attention_precision_matrix.py` (8 shapes × 3 dtype × 4
fidelity × 2 acc × 2 dist, EXCLUSION cells skipped): **272 passed / 112 skipped**;
worst non-degenerate PCC 0.9905 (bf8b/LoFi). `test_regression.py`: the fp32
intermediates fixed **2 of the 9** pre-existing precision misses (the genuine-precision
`×10`-magnitude peaked-softmax cases); the remaining **7** (uniform/negative) are
`max_abs = 1 bf16 ULP`, `ulp_p99=1` — the documented normalized-RMS-on-near-constant-
reference metric artifact, floored by the **bf16 output** quantization (the regression
tests hard-code bf16, so R2's float32 path cannot reach them). Not a bug, never green
in prior phases, outside the registry cartesian (no golden gate).

### [~] Refinement 3 — Speed up the perf-flagged profile (data-movement)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags `1×10×9472×128` (bf16, MHA, self-attn,
`fp32_dest_acc_en=False`, HiFi2) as the mandatory perf target, with
`expected_math_util = 0.35` (≈2.16 ms kernel time) as the goal and a **soft**
`pcc_threshold = 0.997` as the sole gate. Optimize **this** shape. The dominant
lever is data movement: the reader currently issues **one async read + one barrier
+ one push per tile** (the `double_buffer` anti-pattern) even though the KV CBs are
double-buffered, leaving the NoC latency-bound. Apply the relevant patterns from
`ttnn/ttnn/operations/examples/master.md` — `double_buffer` (batch a block of
async reads then one barrier; the CBs are already `KV_DEPTH`-deep) and
`reader_placement` (the split already uses `row_wise=True` — confirm it's optimal
here) look applicable. No SUPPORTED change.

**Verifier notes**: **depends on R2** — the flagged shape's `fp32_dest_acc_en=False`
must be in SUPPORTED (added by R2) before this shape can run at all (it is `xfail`
today). This is the reason R2 precedes R4 in the queue. Headroom is confirmed
qualitatively: the largest measured supported cell (`1×8×4096×128`, 3.47 ms /
110 cores) extrapolates to well above the 2.16 ms target for the ~5× larger
flagged shape, and the shape is data-movement-bound. `/perf-roofline-dm` can
confirm before investing.

**Done when**: measured device-ns improves on the flagged shape (moving
`expected_math_util` toward 0.35), its soft `pcc_threshold=0.997` still holds, the
golden suite is green, and no regression across the config-spanning guard set (one
representative per distinct kernel path × layout × placement — here: mask
none/custom × a small and a large shape, DRAM and L1 output).

**Landed (R3, partial)**: the double_buffer anti-pattern was removed — the reader
now batches a whole KV/Q chunk of async reads behind **one** `noc_async_read_barrier`
(the writer twin batches the whole output chunk behind one `noc_async_write_barrier`),
via a `read_tiles<cb,batch>` / `write_tiles<cb,batch>` helper. Batching is gated on a
straddle-safety predicate (`batch_kv = Skv_t % Skv_chunk_t == 0`, `batch_q = Sq_t %
Sq_chunk_t == 0`) so the multi-page reserve is always slot-aligned in the
`KV_DEPTH`/`OUT_DEPTH`-slot ring (never crosses the CB wrap); partial-chunk shapes keep
the byte-identical per-tile path. Predicates derive from existing CT args — no new
descriptor arg, no hardcoded block/tile literals (DRY). `reader_placement` is already
`row_wise=True` (confirmed optimal). **But device-ns did NOT improve** (baseline
11.06 ms → batched 11.01 ms, within noise). **Ablation (`/perf-measure` no-DM: stub
ALL reader NoC transfers, keep CB reserve/push/barrier)** measures **11.01 ms —
unchanged** → the reader's data movement is **entirely hidden behind compute** by the
existing `KV_DEPTH=2` double-buffer. **The flagged shape is compute-bound, not
data-movement-bound** (current FPU util ≈ 0.07 vs the 0.35 target): reads are off the
critical path, so a DM lever cannot move wall-time here. The batching is **kept** (a
correct, non-regressing removal of the flagged anti-pattern that will surface once
compute no longer dominates), not reverted. Golden **1181 passed / 1088 xfailed**
(0 fail, 0 xpass — no SUPPORTED change); unit **55/55** (incl. non-aligned batched +
partial-chunk per-tile fallback); precision-matrix **272/112**; new guard set
**8/8** (mask none/custom × small/medium × DRAM/L1). The win is gated on the
compute-side work → **R3a** below (which converges with R5).

> **Correction (R3b)**: the `batch=true` path was **parked at its per-tile default**,
> not left active — see R3b's Landed note. The `read_tiles`/`write_tiles` scaffolding
> stays as a live knob for R3a to re-enable + re-measure; the shipped runtime is
> byte-identical to R2's per-tile reader/writer.



### [x] Refinement 3b — Speed up the perf-flagged profile (data-movement) (debug: fix gate violations)

**Goal**: fix the hard violation from Refinement 3 so the completion gate's three bullets hold.

**Verifier notes** (mechanical, from the harness completion gate):

```
Bullet 3 FAIL: REGRESSION — prior-passing golden cells no longer pass (responsible cells 1180/1181). A prior-passing cell that failed, hung, or never ran (suite hung before reaching it) is a regression.
```

**Landed (R3b)**: root-caused the regression to R3's `batch=true` NoC-batching lever
and **parked it at its trivial (per-tile) default** — the shipped reader/writer are
now runtime **byte-identical to the gate-passing R2** state. Evidence chain:
- The full golden suite **passes clean locally** at R3's committed HEAD
  (`1181 passed / 1088 xfailed / 0 failed`, production mode), so the gate's 1-cell
  regression is **not deterministic** on this hardware.
- A `ttnn-static-analyzer` structural review of `batch=true` found **no hazard**:
  the batched reserve/read/barrier/push is byte-identical in L1 layout and
  producer/consumer counts to the per-tile path for every supported shape
  (full-slot, slot-aligned, non-straddling reserves; `get_tile_size ==
  buffer_page_size` for all supported dtypes; no deadlock cycle; balanced counts).
- The lever is **ablation-proven zero-win** (R3: the flagged shape is compute-bound,
  reads hidden behind `KV_DEPTH=2`), so parking it costs no measured perf.
The only plausible mechanism for the intermittent gate failure is a rare bursty-NoC
transient stall on silicon (n≤64 async reads before one barrier) or an infra flake;
neither reproduces here. Parking the knob (compile-time `batch_* = false`) removes
the bursty-read pattern **and** guarantees no regression by construction (== R2).
The `read_tiles<cb,batch>` / `write_tiles<cb,batch>` scaffolding is retained so R3a
flips a one-line predicate to re-enable + re-measure once compute-side work (R5)
exposes the reads on the critical path.

**Verification**: full golden suite **1181 passed / 1088 xfailed / 0 failed**
(ran to completion, no hang — bullets 1 & 3); acceptance + nonaligned + coarse-chunk
+ debug unit **58 passed**; precision baseline + matrix + perf/guard **285 passed /
112 skipped** (flagged-shape soft `pcc≥0.997` held, guard set `pcc≥0.99`) — bullet 2.
No SUPPORTED change (perf refinement).

**Done when**: the gate passes — zero hangs in SUPPORTED, acceptance + refinement tests pass, golden majority with no regression. ✔
### [ ] Refinement 3a — Close the perf win on the compute-bound flagged shape (re-measure DM after compute-side)

**Type**: perf

**Goal** (sharper follow-up from R3's ablation finding): R3 proved — by no-DM
ablation (stubbing every reader NoC transfer leaves device-ns unchanged at 11.0 ms) —
that the flagged `1×10×9472×128` shape is **compute-bound**, so R3's (correct, kept)
reader/writer batching is off the critical path and produced no wall-time win. The
exact next lever is **compute-side**: grow the QKᵀ/PV matmul output subblocks toward
the `fp32_dest_acc_en=False` DEST budget (8 bf16 tiles) and/or coarsen `Sq_chunk_t`/
`Skv_chunk_t` to amortize the ~10 sequential per-chunk helper phases (per-phase
reconfig/init/fill-drain) over more tiles — i.e. lift FPU util from ≈0.07 toward 0.35.
This is exactly **R5's** lever class (`matmul_output_subblock`, `compute_block_size`,
reconfig ablation); R3a is the ordered marker that R5 is the step that closes R3's
target. **After R5 lands, RE-ENABLE + re-measure the R3 DM batching** (parked at its
per-tile default in R3b — flip the reader/writer `batch_q`/`batch_kv` predicates back
to the divisor rules documented in the reader kernel, then re-run the full golden
suite under `--dev` to confirm no intermittent hang before measuring): if the faster
compute exposes the reads (reader becomes the critical path), the re-enabled batching
should now win — tune `KV_DEPTH` / the read-block size; if reads stay hidden, leave the
knob parked (the DM batching is complete-but-dormant, no measured win to bank).

**Verifier notes**: no SUPPORTED change (perf). Do R5 first (or fold R5 into this),
then confirm on the flagged shape that device-ns moves toward the 0.35 util goal with
the soft `pcc_threshold=0.997` holding and the golden suite green.

**Done when**: measured device-ns improves on the flagged shape via the compute-side
lever, and the R3 DM batching is confirmed (still hidden → complete; or exposed →
`KV_DEPTH`/read-block tuned); prior phases green.

### [ ] Refinement 4 — Causal masking (mask_mode = causal)

**Goal**: add `"causal"` to `SUPPORTED["mask_mode"]`, generating the triangular
bias **on-device** (no mask tensor) driven by the `is_causal` compile-time flag.
Two parts, both reachable from phase-0's per-chunk KV loop:
1. **Block-skip** whole future KV chunks (`j·Skv_chunk_t > qc_end` ⇒ the chunk is
   fully masked) — roughly halves the KV work for causal self-attention.
2. **Per-element diagonal mask** on the single straddling KV chunk (the additive
   `−∞` upper-triangular bias), applied to the scaled scores *before* the row-max —
   reuse the existing `has_mask` additive-mask compute path (phase 3), fed by an
   on-device-generated mask instead of a streamed tensor.

Arm `EXCLUSIONS += {"mask_mode": "causal", "attention_kind": "cross"}` (causal
requires `S_q == S_kv`; the rectangular case corresponds to no real workload). The
`is_causal ∧ attn_mask` ValueError is already in `validate()`.

**Verifier notes**: standalone scheme-change (design Lamp 1) — no inventory skill
covers on-device causal masking, so this is verifier-authored. Land **after** R2
so the causal path is validated against the full dtype set already in SUPPORTED
(bf16 first is smallest, but R2's dependency pull-up forces R2 before this; the
causal path is dtype-agnostic and inherits R2's cells). Reuses R1's KV-edge masking
for the straddling-chunk diagonal when `S_q`/`S_kv` are non-aligned. Reference the
design's "Lamps → 1. Causal masking" section and
`tech_reports/FlashAttention/FlashAttention.md` (causal load-balancing).

**Done when**: `mask_mode=causal` self-attention golden cells pass; `causal + cross`
is xfail via the new EXCLUSION; block-skip verified to reduce device-ns on causal
vs an equivalent full-mask custom run; prior phases green.

### [ ] Refinement 5 — Speed up the perf-flagged profile (compute-side)

**Type**: perf

**Goal**: continue optimizing the flagged `1×10×9472×128` profile toward
`expected_math_util = 0.35`, this time with **compute-side** levers distinct from
R3's data-movement work. From `ttnn/ttnn/operations/examples/master.md`:
`matmul_output_subblock` (grow the Q·Kᵀ and P·V output subblocks toward the DEST
budget — with `fp32_dest_acc_en=False` the budget is the full 8 bf16 tiles, so the
`_matmul_subblocks` cap can widen), `compute_block_size` (co-tune the
`Sq_chunk_t`/`Skv_chunk_t` block factors — coarser amortizes per-phase
reconfig/init over more tiles, whole-tile-minimum floor) and its reconfig-ablation
second lever (drop the per-phase data-format reconfig where the format is constant
across the boundary). These are knob-tunes on the block surface `_fit_l1` already
exposes. No SUPPORTED change.

**Verifier notes**: generality is exhausted after R4, so the remaining phases are
all perf (still on the flagged shape — it is the sole flagged case, so R5 uses a
different lever class than R3). Co-tune block size against L1 (the `_fit_l1` budget)
and DEST budget together; the `fp32_dest_acc_en=False` regime (flagged) doubles the
DEST budget vs the phase-0 default, which is exactly what unlocks a wider matmul
subblock. Gate on measurement — if `/perf-roofline-dm` shows the shape is already
data-movement-saturated after R3, fold any residual compute win into R3's guard-set
re-measure instead of shipping a no-op phase.

**Done when**: measured device-ns improves further on the flagged shape beyond R3
(toward the 0.35 util goal), soft `pcc_threshold=0.997` holds, golden green, and no
regression across the config-spanning guard set.
