// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused prep→scan chunked Gated Delta Rule (ONE prim, ONE program, zero DRAM intermediates):
// per head, a dedicated PRODUCER core runs the unchanged prep reader+compute and a writer that
// NoC-writes the 7 computed intermediates (v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv)
// straight into its paired RECEIVER core's CBs via the shipped ready/valid handshake; the
// receiver runs the unchanged scan compute+writer. NP=1 producer and NV=1 (full V) per head.
// Takes prep's inputs, returns scan's outputs — the seven fp32 DRAM tensors of the phased
// hand-off simply never exist. The phased prims (chunk_gdn_phased.hpp) stay in-tree as the
// bit-exact reference: the DRAM round trip they perform is a byte copy, so fused == phased
// bit-for-bit as long as the shared math bodies and CB pack boundaries are untouched.

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

// Union of the prep and scan params (see chunk_gdn_phased.hpp for per-field semantics of the
// prep-side v_flat/HV/qk_flat/Hk/qk_norm/scale block and the scan-side state flags).
struct ChunkGdnFusedParams {
    uint32_t BH;
    uint32_t num_chunks;
    uint32_t chunk_size;
    uint32_t key_dim;
    uint32_t val_dim;
    bool v_flat = false;
    uint32_t HV = 0;
    bool qk_flat = false;
    uint32_t Hk = 0;
    bool qk_norm = false;
    float scale = 1.0f;
    bool has_initial_state = false;
    bool output_final_state = false;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct ChunkGdnFusedInputs {
    Tensor q;                             // [BH, NC, C, K] bf16 (or FLAT [B, T, Hk*K] bf16 when params.qk_flat)
    Tensor k;                             // [BH, NC, C, K] bf16 (or FLAT, as q)
    Tensor v;                             // [BH, NC, C, V] bf16 (or FLAT [B, T, HV*V] bf16 when params.v_flat)
    Tensor g;                             // [BH, NC, C, 1] fp32 (column)
    Tensor beta;                          // [BH, NC, C, 1] fp32 (column)
    Tensor eye_c;                         // [1,1,C,C] fp32
    Tensor tril_c;                        // [1,1,C,C] fp32
    Tensor ones_c;                        // [1,1,C,C] fp32
    Tensor masks_c;                       // [1,1,32,96] fp32 — three 32x32 WY-inverse quadrant masks (Qtl|Qbr|Q10)
    std::optional<Tensor> initial_state;  // [BH, K, V] fp32 or absent (zeros)
};

struct ChunkGdnFusedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ChunkGdnFusedParams&, const ChunkGdnFusedInputs&, std::vector<Tensor>&);
};

struct ChunkGdnFusedOperation {
    using operation_attributes_t = ChunkGdnFusedParams;
    using tensor_args_t = ChunkGdnFusedInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<ChunkGdnFusedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Returns {o [BH,NC,C,V] fp32, final_state [BH,K,V] fp32} — exactly the scan prim's output specs.
// Needs 2*BH cores (one producer + one receiver per head); validate FATALs otherwise, so the
// op-level dispatch must gate on grid size before choosing this path.
std::vector<Tensor> chunk_gdn_fused(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& eye_c,
    const Tensor& tril_c,
    const Tensor& ones_c,
    const Tensor& masks_c,
    const std::optional<Tensor>& initial_state,
    uint32_t chunk_size,
    bool output_final_state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool v_flat = false,
    uint32_t HV = 0,
    bool qk_norm = false,
    float scale = 1.0f,
    bool qk_flat = false,
    uint32_t Hk = 0);

}  // namespace ttnn::prim
