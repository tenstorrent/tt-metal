// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <cmath>

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

SdpaDecodeDeviceOperation::program_factory_t SdpaDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return SdpaDecodeProgramFactory{};
}

void SdpaDecodeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void SdpaDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    bool use_mla = operation_attributes.use_mla.value_or(false);

    if (use_mla) {
        TT_FATAL(
            operation_attributes.head_dim_v.has_value(), "Must provide head_dim_v for multi-latent attention decode");
        TT_FATAL(operation_attributes.is_causal, "Multi-latent attention decode only tested for causal!");
    } else {
        TT_FATAL(tensor_args.v.has_value(), "Must have 3 input tensors and mask");
    }

    std::vector<Tensor> input_tensors{{tensor_args.q}, {tensor_args.k}};
    if (tensor_args.v.has_value()) {
        input_tensors.emplace_back(tensor_args.v.value());
    }
    for (const auto& input_tensor : input_tensors) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to SDPA need to be on device!");
        TT_FATAL(input_tensor.buffer() != nullptr, "Operands to SDPA need to be allocated in buffers on device!");
        TT_FATAL(
            input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::BFLOAT8_B ||
                input_tensor.dtype() == DataType::BFLOAT4_B,
            "Unsupported data type {}.",
            input_tensor.dtype());
    }

    for (size_t i = 1; i < input_tensors.size(); i++) {
        TT_FATAL(input_tensors.at(i).layout() == Layout::TILE, "Inputs to SDPA must be tilized except for Q");
    }

    const auto q_shape = input_tensors.at(0).padded_shape();
    const auto q_shape_unpadded = input_tensors.at(0).logical_shape();
    const auto k_shape = input_tensors.at(1).padded_shape();

    // When using multi-latent attention, the V tensor is the same as K tensor, but of smaller head dimension.
    // For the sake validation, we will use the K tensor shape for V tensor, and validate head_dim_v separately.
    const auto v_shape = use_mla ? input_tensors.at(1).padded_shape() : input_tensors.at(2).padded_shape();

    if (use_mla) {
        // Head dim v validation
        TT_FATAL(
            q_shape[3] == k_shape[3],
            "Head dimension of Q must be equal to head dim of K, got {} and {}",
            q_shape[3],
            k_shape[3]);
        TT_FATAL(
            operation_attributes.head_dim_v.value() <= q_shape[3],
            "Head dimension of V must be less than or equal to head dim of Q, got {} and {}",
            operation_attributes.head_dim_v.value(),
            q_shape[3]);
    }

    // Input 0 must be sharded by height or DRAM interleaved. All other inputs must be in DRAM.
    const auto Q_memcfg = input_tensors.at(0).memory_config();
    if (input_tensors.at(0).is_sharded()) {
        TT_FATAL(
            Q_memcfg.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Q tensor memory layout must be HEIGHT_SHARDED when sharded but got {}",
            Q_memcfg.memory_layout());
    } else {
        TT_FATAL(
            Q_memcfg.buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Q tensor buffer type must be DRAM when not sharded but got {}",
            Q_memcfg.buffer_type());
    }

    for (std::size_t i = 1; i < input_tensors.size(); i++) {
        TT_FATAL(
            input_tensors.at(i).buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Input tensor {} buffer type must be DRAM but got {}",
            i,
            input_tensors.at(i).buffer()->buffer_type());
    }
    // Output memconfig must be height sharded or DRAM
    if (operation_attributes.output_mem_config.is_sharded()) {
        TT_FATAL(
            operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Output memory config layout must be HEIGHT_SHARDED when sharded but got {}",
            operation_attributes.output_mem_config.memory_layout());
    }

    if (!operation_attributes.is_causal) {
        if (tensor_args.attn_mask.has_value()) {
            // Causal attention verification
            const auto& mask_tensor = tensor_args.attn_mask.value();
            const auto mask_shape = mask_tensor.padded_shape();
            const auto mask_shape_unpadded = mask_tensor.logical_shape();

            TT_FATAL(
                mask_shape[2] == q_shape[2],
                "Expect same number of padded heads in mask as in Q, got {} and {}",
                mask_shape[2],
                q_shape[2]);
            TT_FATAL(
                mask_shape_unpadded[2] == q_shape_unpadded[2],
                "Expect same number of heads in mask as in Q, got {} and {}",
                mask_shape_unpadded[2],
                q_shape_unpadded[2]);
            if (!operation_attributes.paged_attention) {
                TT_FATAL(
                    mask_shape[3] == k_shape[2],
                    "Expect same sequence length in mask as in K, got {} and {}",
                    mask_shape[3],
                    k_shape[2]);
            }

            TT_FATAL(operation_attributes.k_chunk_size > 0, "Must provide k_chunk_size if using attention mask!");
            TT_FATAL(
                mask_shape[3] % operation_attributes.k_chunk_size == 0,
                "Mask sequence length must be multiple of chunk size, got: {} and {}",
                mask_shape[3],
                operation_attributes.k_chunk_size);

            TT_FATAL(
                mask_tensor.dtype() == DataType::BFLOAT16 || mask_tensor.dtype() == DataType::BFLOAT8_B ||
                    mask_tensor.dtype() == DataType::BFLOAT4_B,
                "Unsupported data type for mask tensor: {}.",
                mask_tensor.dtype());
        }
    } else {
        // Uncausal attention verification
        TT_FATAL(not tensor_args.attn_mask.has_value(), "Must not have attn_mask tensor for non-causal attention");
    }

    if (operation_attributes.paged_attention) {
        // Paged attention verification
        TT_FATAL(
            !operation_attributes.share_cache.value_or(false), "Share cache feature not supported for paged attention");
        const auto B = q_shape[1];

        if (operation_attributes.is_causal) {
            // Check cur pos tensor for causal mode
            TT_FATAL(
                tensor_args.cur_pos_tensor.has_value(), "Must have cur_pos tensor for paged attention in causal mode");
            const auto& cur_pos_tensor = tensor_args.cur_pos_tensor.value();
            TT_FATAL(
                cur_pos_tensor.dtype() == DataType::INT32,
                "Expect cur_pos to be INT32, got {}",
                cur_pos_tensor.dtype());
            TT_FATAL(
                cur_pos_tensor.layout() == Layout::ROW_MAJOR,
                "Expect cur_pos to be ROW_MAJOR, got {}",
                cur_pos_tensor.layout());
            const auto cur_pos_shape = cur_pos_tensor.padded_shape();

            if (!cur_pos_tensor.is_sharded()) {
                TT_FATAL(
                    cur_pos_shape[-1] == B,
                    "cur_pos must have batch size equal to Q, got {} and {}",
                    cur_pos_shape[0],
                    B);
            }
        }

        TT_FATAL(tensor_args.page_table_tensor.has_value(), "Must have page_table tensor for paged attention");
        const auto& page_table_tensor = tensor_args.page_table_tensor.value();

        if (page_table_tensor.is_sharded()) {
            TT_FATAL(
                page_table_tensor.dtype() == DataType::UINT16,
                "Error: SDPA currently only supports UINT16 datatype for sharded configurations");
        } else {
            TT_FATAL(
                page_table_tensor.dtype() == DataType::INT32, "Error: SDPA currently only supports INT32 datatype");
        }

        TT_FATAL(
            page_table_tensor.layout() == Layout::ROW_MAJOR,
            "Page table tensor layout must be ROW_MAJOR but got {}",
            page_table_tensor.layout());

        const auto page_table_shape = page_table_tensor.padded_shape();

        if (page_table_tensor.is_sharded()) {
            uint32_t num_cores = page_table_tensor.memory_config().shard_spec()->grid.num_cores();
            TT_FATAL(
                page_table_shape[0] / num_cores == B,
                "Page_table must have shard height batch_size {} equal to Q on {} cores",
                B,
                num_cores);
        } else {
            TT_FATAL(page_table_shape[0] == B, "Page_table must have batch size equal to Q");
        }

        TT_FATAL(k_shape[2] == v_shape[2], "K and V must have same block size");
        TT_FATAL(k_shape[3] == v_shape[3] && k_shape[3] == q_shape[3], "Q, K, V must have same hidden size");

        // Validate chunk size for paged version
        // k_chunk_size can also be zero; if k_chunk_size = 0, figure it out in kernels
        TT_FATAL(
            operation_attributes.k_chunk_size % 32 == 0,
            "Chunk size must be multiple of 32, got: {}",
            operation_attributes.k_chunk_size);
        if (!operation_attributes.is_causal) {
            TT_FATAL(operation_attributes.k_chunk_size > 0, "Must provide k_chunk_size if paged and non-causal!");
            TT_FATAL(
                (page_table_shape[1] * k_shape[2]) % operation_attributes.k_chunk_size == 0,
                "K sequence length must be multiple of chunk size, got: {} and {}",
                page_table_shape[1] * k_shape[2],
                operation_attributes.k_chunk_size);
        }
    } else {
        // Unpaged attention verification
        TT_FATAL(
            not tensor_args.page_table_tensor.has_value(), "Must not have page_table tensor for unpaged attention");
        // Assert we are in decode mode if not causal
        // Q: [1, B, NH, D]
        // K: [1, B, S, D]
        // V: [1, B, S, D]

        // Batch must match
        const auto B = q_shape[1];
        if (operation_attributes.share_cache.value_or(false)) {
            TT_FATAL(k_shape[0] == 1, "Share cache expects K to have batch size of 1, but got {}", k_shape[0]);
            TT_FATAL(v_shape[0] == 1, "Share cache expects V to have batch size of 1, but got {}", v_shape[0]);
        } else {
            TT_FATAL(k_shape[0] == B, "K tensor batch size ({}) must equal B ({})", k_shape[0], B);
            TT_FATAL(v_shape[0] == B, "V tensor batch size ({}) must equal B ({})", v_shape[0], B);
        }
        // TT_FATAL(Q_memcfg.shard_spec.value().grid.num_cores() == B, "Q must be height sharded by batch ");

        // Q seqlen must be 1 if we are running decode mode
        TT_FATAL(q_shape[0] == 1, "Q tensor batch size must be 1 for decode mode but got {}", q_shape[0]);

        // Check sequence lengths
        TT_FATAL(
            k_shape[-2] == v_shape[-2],
            "K and V tensors must have the same sequence length. K: {}, V: {}",
            k_shape[-2],
            v_shape[-2]);

        // Validate chunk size for unpaged version
        TT_FATAL(operation_attributes.k_chunk_size > 0, "Must provide k_chunk_size if non-causal!");
        TT_FATAL(
            operation_attributes.k_chunk_size % 32 == 0,
            "Chunk size must be multiple of 32, got: {}",
            operation_attributes.k_chunk_size);
        TT_FATAL(
            k_shape[2] % operation_attributes.k_chunk_size == 0,
            "K sequence length must be multiple of chunk size, got: {} and {}",
            k_shape[2],
            operation_attributes.k_chunk_size);

        // Check hidden size
        const auto D = q_shape[-1];
        TT_FATAL(
            k_shape[-1] == D,
            "K tensor hidden dimension ({}) must equal Q tensor hidden dimension ({})",
            k_shape[-1],
            D);
        TT_FATAL(
            v_shape[-1] == D,
            "V tensor hidden dimension ({}) must equal Q tensor hidden dimension ({})",
            v_shape[-1],
            D);

        // Check valid seqlen
        for (unsigned int cur_pos_val : operation_attributes.cur_pos) {
            TT_FATAL(cur_pos_val < k_shape[-2], "cur_pos must be <= K sequence dim");
        }
    }

    // Check gqa specific validation
    TT_FATAL(
        k_shape[1] == v_shape[1],
        "Flash decode expects K and V to have same number of heads, but got {} and {}",
        k_shape[1],
        v_shape[1]);
    bool is_gqa = (k_shape[1] > 1);
    if (is_gqa) {
        TT_FATAL(!operation_attributes.output_mem_config.is_sharded(), "Sharded output not supported for GQA");
        TT_FATAL(
            input_tensors.at(0).dtype() == DataType::BFLOAT16,
            "GQA expects BFLOAT16 input tensor, but got {}",
            input_tensors.at(0).dtype());
        TT_FATAL(
            q_shape_unpadded[2] % k_shape[1] == 0,
            "GQA expects Q to have a multiple of K heads, but got {} and {}",
            q_shape_unpadded[2],
            k_shape[1]);
    }

    // Check attention sink
    if (tensor_args.attention_sink.has_value()) {
        const auto& attention_sink = tensor_args.attention_sink.value();

        const auto& sink_shape = attention_sink.padded_shape();
        TT_FATAL(sink_shape.size() == 2, "Attention sink must have 2 dimensions");
        TT_FATAL(
            sink_shape[0] == q_shape[2],
            "Attention sink must have the same padded num heads as Q but got {}",
            sink_shape[0]);
        TT_FATAL(
            sink_shape[1] == tt::constants::TILE_WIDTH,
            "Attention sink must be a single tile wide, but got {}",
            sink_shape[1]);
        TT_FATAL(
            attention_sink.dtype() == DataType::BFLOAT16,
            "Attention sink must by a BF16 tensor, but got {}",
            attention_sink.dtype());
        TT_FATAL(
            attention_sink.layout() == Layout::TILE,
            "Attention sink must be in TILE layout, but got {}",
            attention_sink.layout());
        TT_FATAL(
            attention_sink.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Attention sink must be in DRAM memory, but got {}",
            attention_sink.memory_config().buffer_type());
    }
}

TensorSpec SdpaDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.q;
    Layout output_layout = Layout::TILE;
    ttnn::Shape output_shape = input.logical_shape();
    if (input.layout() == Layout::ROW_MAJOR) {
        output_shape[2] = round_up_to_tile(output_shape[2], tt::constants::TILE_HEIGHT);
        output_layout = Layout::ROW_MAJOR;
    }
    bool use_mla = operation_attributes.use_mla.value_or(false);
    if (use_mla) {
        output_shape[3] = operation_attributes.head_dim_v.value();
    }
    return TensorSpec(
        output_shape, TensorLayout(input.dtype(), PageConfig(output_layout), operation_attributes.output_mem_config));
}

Tensor SdpaDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.q.device());
}

tt::stl::hash::hash_t SdpaDecodeDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    bool has_cur_pos = tensor_args.cur_pos_tensor.has_value();
    bool has_attn_mask = tensor_args.attn_mask.has_value();

    return operation::hash_operation<SdpaDecodeDeviceOperation>(
        operation_attributes.scale,
        operation_attributes.output_mem_config,
        operation_attributes.program_config,
        operation_attributes.compute_kernel_config,
        operation_attributes.k_chunk_size,
        operation_attributes.paged_attention,
        operation_attributes.is_causal,
        operation_attributes.use_mla,
        operation_attributes.head_dim_v,
        operation_attributes.sliding_window_size,
        has_attn_mask,
        has_cur_pos,
        tensor_args.q,
        tensor_args.k,
        tensor_args.v,
        // Hash on page_table_tensor to properly size page table CB
        tensor_args.page_table_tensor,
        tensor_args.attention_sink);
}

Tensor sdpa_decode(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const std::optional<const Tensor>& input_tensor_v,
    const std::optional<const Tensor>& cur_pos_tensor,
    const std::optional<const Tensor>& page_table_tensor,
    const std::optional<const Tensor>& attn_mask,
    const std::optional<const Tensor>& attention_sink,
    bool is_causal,
    bool paged_attention,
    const std::vector<uint32_t>& cur_pos,
    std::optional<float> scale,
    std::optional<uint32_t> sliding_window_size,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    uint32_t k_chunk_size,
    std::optional<bool> share_cache,
    std::optional<bool> use_mla,
    std::optional<uint32_t> head_dim_v) {
    using OperationType = SdpaDecodeDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .is_causal = is_causal,
        .paged_attention = paged_attention,
        .cur_pos = cur_pos,
        .scale = scale,
        .sliding_window_size = sliding_window_size,
        .output_mem_config = output_mem_config,
        .program_config = program_config,
        .compute_kernel_config = compute_kernel_config,
        .k_chunk_size = k_chunk_size,
        .share_cache = share_cache,
        .use_mla = use_mla,
        .head_dim_v = head_dim_v,
    };

    auto tensor_args = OperationType::tensor_args_t{
        .q = input_tensor_q,
        .k = input_tensor_k,
        .v = input_tensor_v,
        .cur_pos_tensor = cur_pos_tensor,
        .page_table_tensor = page_table_tensor,
        .attn_mask = attn_mask,
        .attention_sink = attention_sink,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
