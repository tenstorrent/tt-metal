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

### [ ] Refinement 1 — Numerical configurability expansion

**Goal**: add `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` (496 xfail cells across all shapes × mask_mode × scale_mode), expose `compute_kernel_config: ttnn.ComputeKernelConfig` on the entry point, and derive input/probs/output CB formats from the input dtype (intermediate stat/accumulator CBs are already Float32). Cells that fail out of the box land in `EXCLUSIONS`, not their own refinement. Pass condition: zero kernel changes when helpers are wired correctly.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: lands first — Refinement 3 reuses the dtype-driven CB-format derivation and the exposed fidelity/fp32-dest knobs. Mind the documented HiFi4+fp32-DEST-with-bf16 known-bad (matmul_block_helpers.hpp:415). fp32 dtype ≈ doubles Q/K/V/probs CB footprint: re-run the L1 budget; the D=1024 (Dt=32, c=1) corner is at the 1.5 MB boundary and is an acceptable `EXCLUSIONS` candidate.

**Done when**: all dtype-axis xfail cells pass (minus declared EXCLUSIONS); golden run clean.

### [ ] Refinement 2 — GQA / MQA head mapping

**Goal**: add `"gqa"` and `"mqa"` to `SUPPORTED["kv_heads_mode"]` (68 xfail cells incl. GQA/MQA cross-attention and long-context). Compute is head-agnostic; the only structural change is the reader mapping Q head `h` → KV head `h / (H_q / H_kv)` (design line: "GQA/MQA ready"). Pass `H_kv` (or the ratio) as a CT arg; `_validate_shapes` already enforces `H_q % H_kv == 0`.

**Verifier notes**: reader-only change; no skill matches (no new dtype/layout/multicore work). Also un-fails the 4 `test_regression.py::test_gqa_mqa_forward` validation failures. Keep `validate()`'s shape rules unchanged.

**Done when**: every kv_heads_mode=gqa/mqa golden cell passes; gqa_mqa_forward regression tests pass.

### [ ] Refinement 3 — Long-context / near-uniform-attention precision

**Goal**: move the 6 `supported_fail` `numerical-precision` cells (Q1x1x4096x64, Q1x1x8192x64, Q1x4x4096x64 × mask_mode=none × both scale_modes; RMS 0.067–0.158 vs 0.05) and the failing distribution regressions (`uniform_input`, `negative_input` rms 0.09–0.50 incl. one severity=bug at S=512, `long_context_smoke`) into passing. Root cause: near-uniform attention output has tiny stddev while bf16 `cb_probs` quantization noise stays constant — relative RMS blows up with S. Lever: fp32 probs path for rowsum + P@V (fp32 matmul inputs at reduced throughput, gated on compute_kernel_config from Refinement 1) or bf16 probs error compensation in `l`. No SUPPORTED axis changes — these cells are inside SUPPORTED and must not be gated out via EXCLUSIONS or a shape-size tagger.

**Verifier notes**: depends on Refinement 1 (compute_kernel_config + CB-format plumbing). Tolerances live in `eval/golden_tests/.../helpers.py:TOLERANCES` — fix the kernel, don't loosen them.

**Done when**: zero `supported_fail` cells; uniform/negative/long_context regression tests pass at default tolerances.

### [ ] Refinement 4 — Non-tile-aligned shapes

**Goal**: add `"w_non_aligned"` and `"h_non_aligned"` to `SUPPORTED["alignment"]` (40 xfail cells, incl. non-aligned + GQA/MQA/cross combos). Standalone, algorithm-fundamental: padded S_kv columns corrupt rowmax (MAX reduce sees pad) and rowsum (exp(pad − m) ≠ 0), so edge KV tiles need a −∞-style additive pad mask before the max, and padded D / S_q rows need zero-fill on the P@V/output path; non-aligned S_kv also requires lifting `validate()`'s S_kv gate.

**Verifier notes**: last — touches the score path that Refinements 1/3 stabilize, and 9 of the 40 cells also need GQA/MQA (Refinement 2). Not "pad-and-go": the pad mask must enter before the running-max update, exactly like the user mask path (reuse the mask CB plumbing with a generated pad tile).

**Done when**: every alignment-axis xfail cell passes; tile_aligned cells unchanged.
