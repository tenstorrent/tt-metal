# Operation Requirements: scaled_dot_product_attention

## Definition
- **Formula**: `O = softmax(Q @ K^T * scale + mask, dim=-1) @ V`, scale defaults to `1/sqrt(D)`; computed with the Flash-Attention online-softmax recurrence (running max `m`, running sum `l`, fp32 output accumulator; the full S_q × S_kv score matrix is never materialized).
- **PyTorch Reference**:
  ```python
  def torch_sdpa(q, k, v, mask=None, scale=None):
      scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
      if q.shape[1] != k.shape[1]:  # GQA/MQA head broadcast
          r = q.shape[1] // k.shape[1]
          k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
      s = q @ k.transpose(-2, -1) * scale
      if mask is not None:
          s = s + mask
      return torch.softmax(s, -1) @ v
  ```
- **Import Path**: `from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention`
- **Function Signature**: `scaled_dot_product_attention(query: ttnn.Tensor, key: ttnn.Tensor, value: ttnn.Tensor, *, attention_mask: ttnn.Tensor = None, scale: float = None, memory_config: ttnn.MemoryConfig = None) -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred, `[ ]` only when nothing usable was produced.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE] (= full TARGET; SDPA is TILE-only by design)
- **SUPPORTED shape-derived axes**: alignment=tile_aligned; attention_kind ∈ {self, cross}; kv_heads_mode ∈ {mha}
- **SUPPORTED op-specific axes**: mask_mode ∈ {none, causal}, scale_mode ∈ {auto, explicit}
- **Cores**: full grid, embarrassingly parallel over (b, h, q_chunk); interleaved DRAM
- **Compute config**: hard-coded HiFi2 + fp32_dest_acc_en
- **Golden baseline**: 134 / 140 supported cells passing (6 long-context precision misses → Refinement 3); 604 xfail_expected; loud categories all 0 (per verifier_report.json)

### [~] Refinement 1 — Numerical configurability expansion

**Goal**: add `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` (496 xfail cells across all shapes × mask_mode × scale_mode), expose `compute_kernel_config: ttnn.ComputeKernelConfig` on the entry point, and derive input/probs/output CB formats from the input dtype (intermediate stat/accumulator CBs are already Float32). Cells that fail out of the box land in `EXCLUSIONS`, not their own refinement. Pass condition: zero kernel changes when helpers are wired correctly.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: lands first — Refinement 3 reuses the dtype-driven CB-format derivation and the exposed fidelity/fp32-dest knobs. Mind the documented HiFi4+fp32-DEST-with-bf16 known-bad (matmul_block_helpers.hpp:415). fp32 dtype ≈ doubles Q/K/V/probs CB footprint: re-run the L1 budget; the D=1024 (Dt=32, c=1) corner is at the 1.5 MB boundary and is an acceptable `EXCLUSIONS` candidate.

**Done when**: all dtype-axis xfail cells pass (minus declared EXCLUSIONS); golden run clean.

**Outcome (2026-06-10, partial)**: all three dtypes in SUPPORTED; zero kernel changes;
compute_kernel_config exposed (defaults: HiFi2+fp32dest for bf16/bf8b, HiFi4+fp32dest for
fp32); CB formats dtype-derived, intermediates follow fp32_dest_acc_en. 494/496 new dtype
cells pass (412/420 supported incl. Phase 0). Deferred to Refinement 3 (no EXCLUSIONS,
left failing): 2 bf8b Q1x1x8192x64 mask=none cells, RMS 0.165 vs 0.12 — same
near-uniform-attention root cause as Phase 0's 6 bf16 misses. Guards (config-level, not
axes): HiFi4+fp32dest+bf16/bf8b (known-bad #38306, ValueError); fp32/bf8b inputs with
fp32_dest_acc_en=False (16-bit DEST pairing corrupts — probe_008, NotImplementedError).

### [~] Refinement 2 — GQA / MQA head mapping

**Goal**: add `"gqa"` and `"mqa"` to `SUPPORTED["kv_heads_mode"]` (68 xfail cells incl. GQA/MQA cross-attention and long-context). Compute is head-agnostic; the only structural change is the reader mapping Q head `h` → KV head `h / (H_q / H_kv)` (design line: "GQA/MQA ready"). Pass `H_kv` (or the ratio) as a CT arg; `_validate_shapes` already enforces `H_q % H_kv == 0`.

**Verifier notes**: reader-only change; no skill matches (no new dtype/layout/multicore work). Also un-fails the 4 `test_regression.py::test_gqa_mqa_forward` validation failures. Keep `validate()`'s shape rules unchanged.

**Done when**: every kv_heads_mode=gqa/mqa golden cell passes; gqa_mqa_forward regression tests pass.

**Outcome (2026-06-10, partial)**: gqa + mqa in SUPPORTED; reader-only change as
designed (H_kv CT arg, kv_bh = b*H_kv + h/(H/H_kv)); head mapping verified exactly
via deterministic constant-V test. 64/68 GQA/MQA cells pass; all 4 gqa_mqa_forward
regressions pass. 4 deferred cells (no EXCLUSIONS, left failing) are long-context
mask=none precision misses (Q1x4x4096x64 KV1x1 mqa, Q1x8x4096x128 KV1x2 gqa ×
scale_modes; RMS 0.064–0.067 vs 0.05) — identical near-uniform-attention root cause
and magnitude as their MHA siblings already filed under Refinement 3; the head
mapping contributes no extra error. Moved to Refinement 3 inherited list.

### [~] Refinement 3 — Long-context / near-uniform-attention precision

**Goal**: move the 6 `supported_fail` `numerical-precision` cells (Q1x1x4096x64, Q1x1x8192x64, Q1x4x4096x64 × mask_mode=none × both scale_modes; RMS 0.067–0.158 vs 0.05) and the failing distribution regressions (`uniform_input`, `negative_input` rms 0.09–0.50 incl. one severity=bug at S=512, `long_context_smoke`) into passing. Root cause: near-uniform attention output has tiny stddev while bf16 `cb_probs` quantization noise stays constant — relative RMS blows up with S. Lever: fp32 probs path for rowsum + P@V (fp32 matmul inputs at reduced throughput, gated on compute_kernel_config from Refinement 1) or bf16 probs error compensation in `l`. No SUPPORTED axis changes — these cells are inside SUPPORTED and must not be gated out via EXCLUSIONS or a shape-size tagger.

**Verifier notes**: depends on Refinement 1 (compute_kernel_config + CB-format plumbing). Tolerances live in `eval/golden_tests/.../helpers.py:TOLERANCES` — fix the kernel, don't loosen them.

**Inherited from Refinement 2**: also fix the 4 GQA/MQA long-context mask=none cells (Q1x4x4096x64 KV1x1x4096x64 mqa, Q1x8x4096x128 KV1x2x4096x128 gqa × both scale_modes; bf16, RMS 0.064–0.067 vs 0.05) — same root cause and same lever as the 6 MHA bf16 cells above; head mapping verified exact, no GQA-specific error term.

**Inherited from Refinement 1**: also fix the 2 bf8b Q1x1x8192x64 mask=none cells (RMS 0.165 vs 0.12, ulp_p99 9.5e8 — same near-uniform-attention root cause; cb_probs is bf8b for bf8b inputs, so the probs-quantization noise term is larger). Lever: keep cb_probs at Float16_b (or Float32 with the fp32-probs path) regardless of input dtype — Refinement 1 already left `in_fmt` vs `acc_fmt` derivation in the descriptor, so this is one line for the probs CB. fp32 inputs already pass long-context (probs CB Float32 there).

**Done when**: zero `supported_fail` cells; uniform/negative/long_context regression tests pass at default tolerances.

**Outcome (2026-06-10, partial)**: zero `supported_fail` cells — all 12 golden cells
(6 MHA bf16 + 4 GQA/MQA + 2 bf8b) pass at rms 0.008–0.020; `long_context_smoke` and
all GQA/MQA regressions pass. Actual root cause (probes 009–013) was NOT probs
quantization alone: HiFi2 skips the SrcB low-mantissa fidelity phase, so rowsum `l`
and P@V consumed packed probs at different precision (V=ones invariant O==1 off by
11%). Fix: bf16/bf8b default HiFi3 + cb_probs follows acc_fmt (Float32) — zero kernel
changes. Deferred: 9 uniform/negative regression cells (rms 0.04–0.21, all
`ulp_p99=1, median_abs=0`) — output std ≈ 1 bf16 ulp puts default rel-RMS at the
output-quantization floor; static analyzer confirms no structural term remains. See
Refinement 5.

### [x] Refinement 5 — Uniform/negative-input bf16-floor flip-rate reduction

**Goal**: move the 9 deferred `test_uniform_input` / `test_negative_input` regression
cells (S=32–512 bf16; rms 0.04–0.21 vs 0.04, pcc up to 0.9947 vs 0.995) into passing.
All disagreement is single-ulp flips at the bf16 output grid (`ulp_p99=1,
median_abs=0`); rms tolerance requires flip-rate < ~0.2% vs current ~1–2%. Known
residual terms (probes 009–013, static-analyzer report): (a) Phase 12 `O · (1/l)`
runs on FPU — fp32 operands truncate to ~10 mantissa bits (≈0.25 ulp of bf16), and
(b) `recip_tile` SFPU ≈1 ulp fp32. Levers, in order: (1) materialize `inv` to a full
fp32 tile (eltwise Col-bcast copy) and do the final multiply on SFPU (`SfpuMul`,
fp32-exact); (2) verify packer fp32→bf16 rounding mode is RNE (torch's mode);
(3) extra NR iteration on recip (or compute `out = O · (1/l)` with bcast on the
already-HiFi3 FPU using `l` pre-inverted in fp32). No SUPPORTED/EXCLUSIONS changes —
cells stay inside SUPPORTED and failing until fixed.

**Outcome (2026-06-10, full)**: all 9 cells pass; regression suite 39/39, golden
663/0/120 — zero supported_fail. Verifier's residual model was wrong: the dominant
term was neither (a) nor (b) but the descriptor never setting
`UnpackToDestMode::UnpackToDestFp32` — every copy_tile/UnaryBcast of a Float32 CB
silently truncated to fp16, so the O accumulator lost mantissa each KV block
(proved bit-for-bit by host model, probes 028–032). Lever (1) was implemented along
the way (full SFPU stat path) but only halved rms; lever (3) (Newton step on recip,
~3.6e-5 → ~1e-9 rel err) closed the rest. Lever (2): packer is round-half-up, not
RNE — ties differ from torch only at exact bf16 midpoints (measure ~0 for random
data). Flip rates: 0.2–1.0% (pre-fix 2–6%).

### [x] Refinement 4 — Non-tile-aligned shapes

**Goal**: add `"w_non_aligned"` and `"h_non_aligned"` to `SUPPORTED["alignment"]` (40 xfail cells, incl. non-aligned + GQA/MQA/cross combos). Standalone, algorithm-fundamental: padded S_kv columns corrupt rowmax (MAX reduce sees pad) and rowsum (exp(pad − m) ≠ 0), so edge KV tiles need a −∞-style additive pad mask before the max, and padded D / S_q rows need zero-fill on the P@V/output path; non-aligned S_kv also requires lifting `validate()`'s S_kv gate.

**Verifier notes**: last — touches the score path that Refinements 1/3 stabilize, and 9 of the 40 cells also need GQA/MQA (Refinement 2). Not "pad-and-go": the pad mask must enter before the running-max update, exactly like the user mask path (reuse the mask CB plumbing with a generated pad tile).

**Done when**: every alignment-axis xfail cell passes; tile_aligned cells unchanged.

**Outcome (2026-06-10, full)**: w_non_aligned + h_non_aligned in SUPPORTED; golden
test_golden.py 744/744 (was 624 passed / 120 xfail — every alignment cell flipped,
zero xfail left in the universe); unit + regression suites green. Exactly as the
verifier prescribed: reader generates a bf16 pad-mask row (0 valid / −1e9 pad cols,
prepared once, never popped), compute adds it to scale·S(+mask) on the last KV block
via the existing mask plumbing (DestReuseBinary Row/HeldBulk) before the rowmax;
tile counts from padded_shape; S_q/D padding needs no fix (zero-pad benign). S_kv
validate() gate dropped. rel_rms 0.0018–0.0048 bf16 on the canonical shapes.
