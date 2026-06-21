// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <tt-metalium/base_types.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace ttnn::operations::experimental::deltanet {

struct DeltaNetPrefillChunkedParams {
    uint32_t num_heads;     // Hv (one core per head)
    uint32_t k_head_dim;    // Dk
    uint32_t v_head_dim;    // Dv
    uint32_t chunk;         // C (== 32)
    uint32_t n_chunks;      // Sp / C
    uint32_t seq_len;       // S (real, unpadded)
    tt::tt_metal::MemoryConfig output_memory_config;
};

// Factored chunked gated-delta-rule prefill. All decay scalings are folded into the
// keys/queries on host (precision-safe, validated PCC≈1.0). Per-head data is laid out
// row-major as [Hv*Sp, D] so head h, chunk c occupies tile-row (h*n_chunks + c); k_T /
// KiT are [Hv*Dk, Sp] so head h occupies row-block [h*Dk : (h+1)*Dk], chunk c col-tile c.
// Constant masks (identity, tril) + RMS scaler/eps are generated in the reader kernel.
struct DeltaNetPrefillChunkedInputs {
    const Tensor& k;                // [Hv*Sp, Dk]  raw l2-normed key (head-expanded)
    const Tensor& q;                // [Hv*Sp, Dk]  raw l2-normed q-scaled query
    const Tensor& v;                // [Hv*Sp, Dv]
    const Tensor& z;                // [Hv*Sp, Dv]  gate
    const Tensor& Kdec;             // [Hv*Sp, Dk]  (beta*d) * k
    const Tensor& KiT;              // [Hv*Dk, Sp]  transpose of (1/d * k)
    const Tensor& Qd;               // [Hv*Sp, Dk]  d * q
    const Tensor& dcol;             // [Hv*Sp, Dv]  d (chunk-cumprod decay), full Dv width
    const Tensor& betacol;          // [Hv*Sp, Dv]  beta, full Dv width
    const Tensor& dlast;            // [Hv*Sp, Dv]  d_last broadcast (chunk's last d), full width
    const Tensor& recurrent_state;  // [1, Hv, Dk, Dv]  entering state (zeros for fresh)
    const Tensor& norm_weight;      // [1,1,1,Dv]
};

}  // namespace ttnn::operations::experimental::deltanet
