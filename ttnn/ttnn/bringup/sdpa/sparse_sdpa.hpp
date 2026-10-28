// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <cstdint>
#include <optional>

namespace ttnn::transformer::bringup {

enum class SparseKVFormat : uint8_t {
    BF16,
    FP8_E4M3,
    SCALED_FP8,
    // Bring-up fork option: bfloat4_b row records (K_DIM / 16 shared exponents, then K_DIM / 2 bytes of sign-magnitude
    // nibbles, value 2i in the low nibble of byte i: the bytes of a bfloat4_b tile row) in a uint8 ROW_MAJOR kv, plus a
    // bf16 kv_window for the first k_chunk of the index list (see sparse_sdpa below).
    ROW_BFP4,
};

// Sparse MLA prefill (DeepSeek DSA), Blackhole single-chip.
//   q       [1, H, S, K_DIM] bf16 or fp8_e4m3 ROW_MAJOR  (H = head count, any multiple of 32; K_DIM = head
//                                                          dim, e.g. 576)
//   kv      [1, 1, T, K_DIM] bf16 or fp8_e4m3 ROW_MAJOR  (K = full K_DIM, V = kv[..., :v_dim]; fp8 halves
//                                                          the K-gather bytes, tilized in-op to bfp8_b), or a
//           packed scaled-FP8 row described below
//   indices [1, 1, S, TOPK] uint32 ROW_MAJOR (0xFFFFFFFF = masked; sentinels are a contiguous tail)
//   v_dim   width of V (leading v_dim cols of the K_DIM-wide cache); the output width.
//   kv_format explicitly identifies the KV representation and must agree with its dtype and logical width.
// Returns out [1, H, S, v_dim] ROW_MAJOR; the output dtype MATCHES q (bf16 q -> bf16 out, fp8 q -> fp8 out).
// (K_DIM is taken from q/kv; scale defaults to K_DIM**-0.5.)
//
// cache_batch_idx: when set, kv is a shared [B, 1, T, K_DIM] cache and this selects the batch slot to attend
// to (indices are page ids within that slot). kv may then also be ND-sharded across DRAM banks. The value is
// a dynamic runtime arg, so changing the slot (or the cache length T) does not recompile the kernels.
//
// block_cyclic_sp_axis / block_cyclic_chunk_local: when both set, `indices` are NATURAL token positions but kv
// is stored block-cyclic across an SP-sharded cache (the DeepSeek chunked-prefill KVPE cache). The gather
// kernels remap each index natural -> physical page on the fly, so the caller does NOT need to reorder kv back
// to natural order. Both must be set together.
//   block_cyclic_sp_axis    : the MESH axis the cache was striped over. `sp` is read from mesh shape on that
//                             axis (the op derives it, so a caller cannot pass an sp inconsistent with the
//                             device); T % sp == 0 required.
//   block_cyclic_chunk_local: the per-shard chunk length (chunk_size_global / sp). Cross-checked at the entry
//                             against q's per-chip seq length: must equal q_isl or tp*q_isl (tp = mesh/sp),
//                             the only two values it can legally take (post-reshard q is sliced by tp).
//   block_cyclic_cache_tp_sharded : true = the cache is striped across ALL sp*tp devices (linear chip = sp_coord*tp
//                             + tp_coord), so stripes = sp*tp and per-stripe chunk = chunk_local/tp.
//
// high_precision (bring-up fork option, default false = the source program): keep the flash running state that is
// L1-accumulated across k_chunks (output and row-sum ping-pong CBs) in Float32 instead of bf16, and compute the
// softmax exp exactly (the source always uses the fast approximate exp there; the scale then multiplies in as a
// bf16 immediate, rounded to nearest). Needs fp32_dest_acc_en, bf16 q and a BF16 kv. An attention_sink is
// supported (its exp uses the same rounded-to-nearest bf16 scale).
//
// fp32_output (bring-up fork option, default false = output in q's dtype; needs high_precision): return the
// normalized output in Float32 (the fp32 DEST value, untilized exactly) instead of rounding it to bf16.
//
// fp32_scores (bring-up fork option, default false = bf16 scores / probabilities; needs high_precision and
// fp32_output): keep the scores Q@Kᵀ and the softmax probabilities (the operand of probs@V) in a Float32 CB instead
// of rounding them to bf16, and accept a FLOAT32 q (tilized to a Float32 CB). The FPU reads Float32 operands at TF32
// by truncation (a coherent bias), so every Float32 value it reads later is rounded to TF32 to nearest-even in DEST
// (SFPU, pack thread) before it is packed: the scores, the probabilities, the running output (probs@V), the row-sum
// (summed in DEST when a k_chunk's key tiles fit DEST) and 1/sum. A Float32 q is read as given (round it first). The
// running max stays bf16 (it is subtracted from the scores and the sink alike).
//
// bf16_probs (bring-up fork option, default false = TF32 probabilities and a bf16 running max; needs fp32_scores and
// a bf16 q): DeepSeek's sparse_attn kernel precision. The running max is Float32 (the max score's probability is
// exp(0) = 1); the row-sum is summed from the TF32 probabilities; then the probabilities are rounded to bf16
// (nearest-even) in DEST, in place, before probs@V (the Float32 CB holds bf16 values, which the FPU reads exactly).
// fp32_output may be false: the output is then rounded to bf16 by the packer (the kernel's bf16 output). The flash
// correction (more than one k_chunk) stays bf16.
//
// kv_format ROW_BFP4 (bring-up fork option, deepseek_v41_flash_d_p 4-bit compressed KV): kv is [B,1,T,320] uint8
// ROW_MAJOR bfloat4_b row records (K_DIM 512: the 32 shared exponents twice, then 256 bytes of nibbles; written by
// ttnn.bringup.ckv_bfp4) and kv_window [1,1,Tw,K_DIM] bf16 ROW_MAJOR. The index list is [window k_chunk | top-k]: its
// first k_chunk ids index kv_window (valid ids first, a sentinel tail), the rest index kv (their own sentinel tail;
// none valid is allowed). A top-k chunk's rows are gathered by plain NoC reads into a row-major bfloat4_b block
// (exponents | nibbles) that the unpacker turns into bfloat4_b K tiles (one UNPACR a face row, datacopy, pack): no
// data-movement RISC byte work, no scale multiply; the window chunk is tilized in bf16 as with BF16. bf16 q only; K_DIM
// 512; not with block_cyclic or high_precision.
//
// attention_sink: optional [1,1,1,H] unpadded interleaved ROW_MAJOR BF16 tensor in DRAM.
// Like classic SDPA, the sink is multiplied by scale and contributes only to the softmax denominator.
// DeepSeek-V4 stores sinks in the already-scaled logit domain: pass model_sink / scale (scale != 0),
// reshaped to [1,1,1,H], so the denominator receives exp(model_sink), not exp(scale * model_sink).
//
// Producer preconditions (NOT validated per-element): sentinels are a contiguous tail, every row has >= 1
// valid key, and all non-sentinel indices are < T.
//
// Scaled FP8 cache: pass SparseKVFormat::SCALED_FP8 and one byte-addressed fp8_e4m3 tensor with each token row packed
// as [v_dim FP8 latent bytes | v_dim/128 FP32 scales | K_DIM-v_dim BF16 RoPE values]. For the DSA 512+64 geometry this
// is 656 logical bytes. Sparse gather writes each selected packed row once into a shared L1 allocation; compute reads
// aliased FP8-latent and BF16-RoPE views and applies the scales while tilizing.
ttnn::Tensor sparse_sdpa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& indices,
    uint32_t v_dim,
    SparseKVFormat kv_format,
    std::optional<float> scale = std::nullopt,
    uint32_t k_chunk_size = 128,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<uint32_t> cache_batch_idx = std::nullopt,
    std::optional<uint32_t> block_cyclic_sp_axis = std::nullopt,
    std::optional<uint32_t> block_cyclic_chunk_local = std::nullopt,
    bool block_cyclic_cache_tp_sharded = false,
    const std::optional<ttnn::Tensor>& attention_sink = std::nullopt,
    bool high_precision = false,
    bool fp32_output = false,
    bool fp32_scores = false,
    bool bf16_probs = false,
    const std::optional<ttnn::Tensor>& kv_window = std::nullopt);

}  // namespace ttnn::transformer::bringup
