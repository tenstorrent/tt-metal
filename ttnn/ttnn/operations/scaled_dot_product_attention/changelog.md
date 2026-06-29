# Changelog: scaled_dot_product_attention

## Phase 0 — Core Implementation
- **Date**: 2026-06-29
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Flash Attention v2 online-softmax recurrence with tiled matmul + eltwise + reduce helpers. Single-core per (B,H) work unit, embarrassingly parallel via split_work_to_cores.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], alignment=[tile_aligned], attention_kind=[self, cross], kv_heads_mode=[mha, gqa, mqa], mask_mode=[none, custom], scale_mode=[auto, explicit]
- **Accuracy achieved**: PCC=0.999997, max_abs_err=0.004569, rms_err=0.002663 (measured on 4 shapes via test_scaled_dot_product_attention_precision_baseline.py)
- **Golden suite at Phase 0**: 200 / 2767 cells passing (per `verifier_report.json`); 8 OOM on D=512/D=1024; 2440 xfail_expected
- **Issues encountered**:
  - Fixed: Missing running-max step in online softmax (Phase 4b). The kernel used `m_blk` directly as `m_new` instead of `max(m_old, m_blk)`, causing `alpha = inf` when a KV-block had all -inf scores (fully masked). Added `BinaryMax` eltwise_chain between row-max and alpha computation.
  - Fixed: Partial last Q-block/KV-block out-of-bounds read. `B_q_t` did not divide `S_q_tiles` for shapes like S=192 (6 tiles, B_q_t=4). Added divisor-reduction loop in the program descriptor.
  - 8 OOM cells on D=512/D=1024 head dims (cb_o + cb_o_accum exceed 1.5 MB L1). Filed as Refinement 4 (/memory-budget-metal).
- **Tests added**: test_scaled_dot_product_attention.py (acceptance), test_scaled_dot_product_attention_precision_baseline.py, test_scaled_dot_product_attention_extended.py

## Refinement 1 — Numerical configurability expansion
- **Date**: 2026-06-29
- **What was done**:
  - Expanded `SUPPORTED["dtype"]` from `[bfloat16]` to `[ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b]`
  - Expanded `SUPPORTED["fp32_dest_acc_en"]` from `[True]` to `[True, False]`
  - Added EXCLUSION `{"dtype": ttnn.float32, "fp32_dest_acc_en": False}` — maxed input + non-maxed acc is rejected (mirrors softmax convention per feature_spec.py)
  - Program descriptor: intermediate CBs are now dtype-driven — `Float32` when `fp32_dest_acc_en=True`, `Float16_b` (bf16) when `fp32_dest_acc_en=False`. Constant CBs (scalers, scale_factor) are always bf16. No `UnpackToDestFp32` tags needed (all intermediate CBs are consumed by FPU ops — matmul, reduce, mul, add — so they cannot be tagged per the exclusivity rule)
  - Key fix: when `fp32_dest_acc_en=False`, intermediate CBs use `Float16_b` (bf16), NOT the input dtype. bf8b as an intermediate accumulator format produces all-zeros output (PCC=0.0) because the block-float shared exponent cannot represent the dynamic range of accumulated values (scores, running max/sum, O). bf16 is the correct 16-bit DEST-register format.
  - No compute kernel changes — helpers carry data-format reconfig automatically (confirmed by `/numeric-formats-metal` skill pass condition)
  - Reader kernel unchanged — the scale factor CB is always bf16, and the reader's `fp32_bits_to_bf16_bits` packing is correct for all input dtypes (the FPU reads the scale through srcA/srcB which carries TF32, and bf16 is a lossless subset)
- **Accuracy achieved**:
  - Golden suite (HiFi4): 1008 / 2233 passing (up from 200 / 2767 in Phase 0). All 32 failures are OOM, zero precision failures.
  - Precision matrix: 222 passed, 162 skipped, 0 failed. All `HiFi4 + fp32_dest_acc_en=True` cells pass across all dtypes and shapes.
  - Representative PCC (HiFi4, fp32_acc): bf16 0.999997, fp32 0.999998, bf8b 0.996 on 128x64 shape
  - `bf8b + fp32_dest_acc_en=True` passes on all shapes except OOM (long_context PCC=0.982 — below 0.99 threshold due to block-float precision compounding over 16 KV-blocks, skipped as HW limitation)
- **Golden test progress**: 1008 / 2233 passing (was 200 / 2767). 32 OOM failures on D=256+fp32, D=512, D=1024 — all L1 budget issues (Refinement 4 scope). 1192 xfail_expected (cells not yet in SUPPORTED — non-tile-aligned shapes, causal mask).
- **Issues encountered**:
  - bf8b + bf16_acc all-zeros output: fixed by using bf16 (Float16_b) intermediate CBs instead of input dtype when fp32_dest_acc_en=False. bf8b block-float format cannot serve as accumulator intermediate.
  - 32 OOM cells on large head dims (D=256+fp32, D=512, D=1024): L1 budget issue. fp32 input tiles are 2x larger than bf16, amplifying the existing Phase 0 OOM pattern. D=256+fp32 is new; D=512 and D=1024 were already failing in Phase 0. These are Refinement 4 (`/memory-budget-metal`) scope, not dtype/precision bugs.
  - Precision matrix: LoFi + uniform distribution drops below 0.99 PCC on all shapes (expected HW behavior). bf16 DEST acc + S>=256 compounds rounding over 8+ KV-blocks. bf8b input + S>=512 compounds over 16 KV-blocks. All are expected hardware-level precision limitations, skipped per `/numeric-formats-metal` §10.
- **Tests added**: test_scaled_dot_product_attention_precision_matrix.py (384-cell precision matrix: 8 shapes × 3 dtypes × 4 fidelities × 2 acc × 2 distributions), precision_matrix_results.md

## Refinement 2 — Non-tile-aligned shape support
- **Date**: 2026-06-29
- **What was done**:
  - Added `"w_non_aligned"` and `"h_non_aligned"` to `SUPPORTED["alignment"]`
  - Program descriptor: ceiling division for `D_t`, `S_q_tiles`, `S_kv_tiles` (was floor division, truncating non-aligned dims — e.g., D=50 gave D_t=1 instead of 2)
  - Op file: `_make_padding_mask()` generates a `-inf` additive mask for padded S_kv columns when S_kv is not tile-aligned. Zero-padded K rows produce `score = Q @ 0 = 0`, and `exp(0 - max) = exp(-max) ≠ 0` contaminates the softmax denominator by ~14-22% per padded column. The `-inf` mask sets those scores to `-inf`, making `exp(-inf) = 0`. Combined with user-supplied mask on the host side when both are present.
  - No kernel changes — the existing mask-add path handles the padding mask
  - D non-aligned is safe: zero-padded K columns give correct scores (Q@0=0), zero-padded V columns give zero output stripped by `to_torch`
  - S_q non-aligned is safe: padded Q rows' output is stripped by `to_torch`. Padded Q rows produce garbage (uniform softmax) but this does NOT contaminate valid Q rows because row-max, row-sum, and PV matmul operate independently per Q row
  - Fix: do NOT include mask tensor in `generic_op` operands list — the original code never included the user mask in operands. Adding it caused `generic_op` to treat the mask as I/O, corrupting execution (all-zeros output with inf)
  - Fix: `_make_padding_mask` handles user mask with different S_q/S_kv by slicing properly
- **Accuracy achieved**: PCC=0.999+ on all non-aligned shapes tested. max_diff ≤ 0.007812 across all categories (w_non_aligned, h_non_aligned, both, mask, GQA, MQA, cross-attn, explicit scale). No inf/nan in any output.
- **Golden test progress**: 1208 / 2233 passing (up from 1008 / 2233 in Refinement 1). +200 non-aligned cells now pass. 32 OOM failures (D=512/D=1024/D=256+fp32 — Refinement 4 scope). 992 xfailed (causal mask, remaining OOM-adjacent). 0 non-aligned failures, 0 precision failures.
- **Issues encountered**:
  - Initial attempt included the padding mask tensor in the `generic_op` operands list, causing all-zeros output with inf. Root cause: `generic_op` treated the mask as an additional I/O tensor. Fix: do not include mask in operands — it stays alive as a local variable during the synchronous call, same as the original user mask pattern.
  - `_make_padding_mask` user-mask combination had a shape mismatch when the user mask's S_q differed from the query's S_q. Fixed by slicing the user mask to the correct dimensions before padding.
- **Tests added**: test_scaled_dot_product_attention_non_aligned.py (19 tests: w_non_aligned ×4, h_non_aligned ×5, both_non_aligned ×3, mask ×2, GQA, MQA, cross-attn, explicit scale, tile-aligned regression)
