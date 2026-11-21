// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include <optional>
#include <variant>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/program.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::prim {

// Sequence geometry shared by the program factory and the program hash. q_num_chunks, gqa_pack,
// writer_gqa_pack_sqt, kv_retain, kernel_padding_mask and k_partial_col pick a kernel variant or the core split
// and are part of the cache key; the rest reaches the kernels as runtime arguments, so one cached program serves
// every Q and K length.
struct SDPASeqGeometry {
    uint32_t gqa_pack = 1;             // Q heads per scheduled head (GQA Q-head crossing; 1 = none)
    uint32_t NQH = 0;                  // scheduled Q heads: logical Q heads / gqa_pack
    uint32_t q_tiles = 0;              // logical Q length in tiles, before crossing (the output's row tiles)
    uint32_t writer_gqa_pack_sqt = 1;  // writer arg 29: logical Q tiles per head when crossing writes concat heads
    uint32_t Sq = 0;                   // scheduled Q length: logical Sq * gqa_pack
    uint32_t Sk = 0;                   // includes the chunked prefix
    uint32_t padded_Sq = 0;
    uint32_t padded_Sk = 0;
    uint32_t Sqt = 0;
    uint32_t Skt = 0;
    uint32_t valid_Sqt = 0;
    uint32_t valid_Skt = 0;
    uint32_t Sq_chunk_t = 0;
    uint32_t Sk_chunk_t = 0;
    uint32_t q_num_chunks = 0;
    uint32_t k_num_chunks = 0;
    bool use_padded_mask = false;      // non-causal with Sq or Sk not chunk aligned
    bool kernel_padding_mask = false;  // the kernels synthesize that padding mask (no provided mask, not windowed)
    uint32_t k_partial_col = 0;        // streaming kernels: Sk % TILE for the partial-tile mask
    uint32_t kv_retain = 0;            // reader arg 38 / compute arg 29: Q heads per K/V retention key (0 = off)
};

SDPASeqGeometry sdpa_seq_geometry(const SDPAParams& attrs, const SDPAInputs& tensors);
bool sdpa_causal_pairs_uniform(const SDPASeqGeometry& geometry);

struct SDPAOperation {
    using operation_attributes_t = SDPAParams;
    using tensor_args_t = SDPAInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct SDPAProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<SDPAProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& attrs, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor);
};

Tensor sdpa(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const std::optional<Tensor>& input_tensor_v,
    const std::optional<Tensor>& attn_mask,
    const std::optional<Tensor>& page_table_tensor,
    const std::optional<Tensor>& attention_sink,
    bool is_causal,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    std::optional<int64_t> chunk_start_idx,
    const std::optional<Tensor>& chunk_start_idx_tensor,
    bool use_mla,
    std::optional<uint32_t> head_dim_v,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<Tensor>& cu_window_seqlens = std::nullopt,
    uint32_t windowed_q_token_offset = 0,
    const std::optional<Tensor>& windowed_q_token_offset_tensor = std::nullopt,
    std::optional<ttnn::operations::transformer::PagedCacheGeometryOverride> paged_cache_geometry = std::nullopt,
    bool output_concat_heads = false,
    const std::optional<Tensor>& attn_mask_block_map = std::nullopt);

}  // namespace ttnn::prim
