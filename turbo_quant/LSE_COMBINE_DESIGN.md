# LSE-Aware Hybrid SDPA Combine — Implementation Plan

**Goal:** make the sliding-window hybrid actually work by combining old-TQ
SDPA output and recent-BFP8 SDPA output via *online softmax* with LSE,
instead of the failed "ring-only" approximation from `3dcc01b1db9` that
discarded old positions entirely.

**Why mandatory:** the W=64 ring-only test landed at 24 % top-1 vs Llama
baseline 97.3 %. Llama doesn't tolerate sliding-window attention; old
positions must contribute via softmax merge.

---

## Combine math (host side)

Each SDPA call returns `(out_X, LSE_X)` where:
- `out_X` = normalized output for region X (what kernels return today)
- `LSE_X` = log-sum-exp = `max_X + log(sum_X)`, where `max_X` and `sum_X`
  are the running max and sum of `exp((s − max) · scale)` from the
  internal online softmax.

Combine:
```
LSE_combined = max(LSE_a, LSE_b) + log(1 + exp(-|LSE_a − LSE_b|))
weight_a     = exp(LSE_a − LSE_combined)
weight_b     = exp(LSE_b − LSE_combined)
out_combined = out_a · weight_a + out_b · weight_b
```

Provably equivalent to running one big SDPA over the union of regions.
Per-layer cost: ~4 small TTNN ops on tensors of shape `[B, NQH, 1]`
(LSE) and `[B, NQH, 1, DH]` (out). Negligible vs the SDPA itself.

---

## Implementation steps

### 1. Fused TQ SDPA: expose LSE  (~½ day)

**File:** `ttnn/.../turbo_quant/sdpa/kernels/compute/sdpa_tq_decode.cpp`

**Spot:** lines 1055-1057 — current finalize:
```cpp
recip_block_inplace(alias_prev_sum, Sq_chunk_t);   // 1/sum
mul_block_bcast_cols<...>(alias_mm2_prev_out, alias_prev_sum, cb_out); // out = unnormalized · 1/sum
```

Insert BEFORE the recip:
```cpp
if constexpr (return_lse) {
    // Compute LSE = max + log(sum) and pack to cb_lse_out (new CB).
    //   - alias_prev_sum currently holds sum after the zero-position correction
    //   - alias_prev_max holds the final max
    // Layout: 1 tile per Sq_chunk_t row — broadcast across head_dim columns.
    cb_reserve_back(cb_lse_out, Sq_chunk_t);
    cb_wait_front(alias_prev_max, Sq_chunk_t);
    for (uint32_t i = 0; i < Sq_chunk_t; ++i) {
        tile_regs_acquire();
        copy_tile_to_dst_init_short(alias_prev_sum);
        copy_tile(alias_prev_sum, i, 0);                       // DST 0 = sum
        log_tile_init();
        log_tile(0);                                           // DST 0 = log(sum)
        copy_tile_to_dst_init_short(alias_prev_max);
        copy_tile(alias_prev_max, i, 1);                       // DST 1 = max
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);                              // DST 0 = LSE
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_lse_out);
        pack_tile(0, cb_lse_out);
        tile_regs_release();
    }
    cb_push_back(cb_lse_out, Sq_chunk_t);
}
// ... existing finalize unchanged ...
```

`log_tile`/`log_tile_init` exist in `compute_kernel_api/eltwise_unary/exp.h`
(or similar — verify with `grep`).

**Add compile-time arg:** slot after `norms_are_bfp8` (TQ_BASE+3+num_levels).

### 2. Fused TQ writer + program factory  (~½ day)

**Writer** (`writer_tq_decode.cpp`):
- New runtime arg: `lse_addr` (output buffer address)
- After existing output write: read `cb_lse_out`, write to `lse_addr`
  with a small TensorAccessor (1 tile per (B, NQH) tuple).

**Program factory** (`sdpa_tq_program_factory.cpp`):
- Allocate `cb_lse_out` (CBIndex::c_32, size = `Sq_chunk_t * im_tile_size`).
- Add `return_lse` boolean to `operation_attributes_t`.
- When `return_lse`: pre-allocate output_lse tensor of shape
  `[B, NQH, 1, 32]` BF16 (or `[B, NQH, 1, 1]` if writer can pack a single
  bf16 — but TILE_LAYOUT pads so a full tile is simpler).
- Pass `output_lse.buffer()->address()` as new writer runtime arg.

**Device op** (`sdpa_tq_device_operation.{cpp,hpp}`):
- `tensor_return_value_t` becomes `std::variant<Tensor, std::tuple<Tensor, Tensor>>`
  or simply always returns `std::tuple<Tensor, Tensor>` with the LSE
  tensor empty when `return_lse=false`.
- Update `compute_output_specs` and `create_output_tensors`.

**Python wrapper** (`turbo_quant.cpp` + nanobind):
- New kwarg `return_lse: bool = False`.
- Return type changes to optional tuple.

### 3. Standard SDPA decode: expose LSE  (~1 day, riskier)

**File:** `ttnn/.../transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp`
plus its program factory and device op.

Same conceptual change as #1 but in a kernel shared by the rest of the
model. Must be:
- Default-off (`return_lse: bool = False` in the op kwargs).
- Gated so non-LSE callers see no change.
- Carefully tested — this kernel has many use cases.

The compute kernel's online softmax state lives in `cb_max_A/B` and
`cb_sum_A/B` (alternating per chunk). At end of chunk loop, alias to the
"last written" buffer holds the final state. Add a similar log+add+pack
block before the final divide.

Writer + program factory + device op + Python: same plumbing as #1.

### 4. attention.py: hybrid path with combine  (~3 hr)

In the `--tq-recent-window > 0` branch of `forward_decode`:

```python
W = turbo_quant_cache.recent_window
old_pos = current_pos − W                # shape [B], int32
ring_pos = ttnn.clamp(current_pos, max=W − 1)

# Old positions: TQ FD kernel with adjusted cur_pos
out_old, lse_old = turbo_quant_cache.fused_sdpa_decode(
    q_pre_rotated, layer_idx=layer_idx,
    current_pos=old_pos,
    scale=self.scale, page_table=tq_page_table,
    return_lse=True,
)

# Recent positions: standard SDPA over BFP8 ring
out_new, lse_new = turbo_quant_cache.sliding_window_sdpa_decode(
    q_pre_rotated, layer_idx=layer_idx,
    current_pos=ring_pos,
    scale=self.scale,
    return_lse=True,
)

# Online softmax combine (host-level, ~4 ops on small tensors)
lse_max = ttnn.maximum(lse_old, lse_new)
diff_old = ttnn.exp(ttnn.subtract(lse_old, lse_max))
diff_new = ttnn.exp(ttnn.subtract(lse_new, lse_max))
denom = ttnn.add(diff_old, diff_new)
weight_old = ttnn.divide(diff_old, denom)
weight_new = ttnn.divide(diff_new, denom)
attn_out = ttnn.add(
    ttnn.multiply(out_old, weight_old),  # broadcast on head_dim
    ttnn.multiply(out_new, weight_new),
)
```

Edge case: `current_pos < W` — only ring valid. Just take `out_new`.

### 5. Validate (~½ day)

- Smoke: 32-layer "Paris" with `--tq-full-dequant --tq-recent-window 64`.
  Expect coherent answer (same as Track A).
- Accuracy: `eval_token_accuracy` at W ∈ {32, 64, 128, 256}. Expect
  monotonic improvement up to a saturation point.
- Comparison: best W vs BFP8 baseline (97.3 / 100 %) and pure Track A
  (86.9 / 97.1 %).

---

## Risk register

| Risk | Mitigation |
|---|---|
| `log_tile` SFPU not exact for very small `sum` (FP32 underflow) | Use a small ε floor on sum: `log(max(sum, 1e−10))` |
| Two SDPA calls double the per-layer dispatch cost | Each call is small (W ≤ 256 vs 1024+ tokens); host combine is small. Per-step latency may go up ~30 % but this is acceptable since it replaces only the SDPA portion |
| Standard SDPA kernel modification breaks model | Default-off flag; run existing model regressions before merging |
| Numerical drift across log/exp roundtrips at BF16 | LSE tolerates BF16 well (it's already log of a sum); known-stable trick |

---

## Estimated total: 2-3 days

| Step | Effort |
|---|---|
| 1. Fused TQ kernel LSE | ½ day |
| 2. Fused TQ writer + program factory + device op | ½ day |
| 3. Standard SDPA kernel LSE + plumbing | 1 day |
| 4. attention.py hybrid + combine | ¼ day |
| 5. Validation + W-sweep | ½ day |
| Buffer for surprises | ¼ day |

---

## Success criteria

- Top-1 at W=64 ≥ 92 % (5 pp better than pure Track A 86.9 %).
- Top-1 at W=128-256 ≥ 95 % (close to BFP8 baseline).
- E2E ms/tok within 1.4× of pure Track A (extra cost = 2nd SDPA + combine).
- KV memory ratio still ≤ 0.80× of BFP8 baseline.
