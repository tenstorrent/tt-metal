// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "group_attn_matmul_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

GroupAttnMatmulDeviceOperation::program_factory_t GroupAttnMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return GroupAttnMatmulProgramFactory{};
}

void GroupAttnMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void GroupAttnMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // input_a: [q_len, q_heads, batch, head_dim]
    // input_b: [batch, kv_heads, head_dim, kv_len]
    // intermediate: [q_heads, batch, batch, kv_len]
    // output: [q_len, q_heads, batch, kv_len]

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    TT_FATAL(
        (input_tensor_a.layout() == Layout::TILE && input_tensor_b.layout() == Layout::TILE),
        "Inputs to matmul must be tilized");

    // TODO: Uplift to support BFLOAT8_B and mixed precision
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE,
        "Operands to matmul need to be on device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr,
        "Operands to matmul need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Operands to matmul need to be on the same device!");

    const auto& ashape = input_tensor_a.padded_shape();
    const auto& bshape = input_tensor_b.padded_shape();
    TT_FATAL((ashape[0] == 1), "Input q_len must be 1!");
    TT_FATAL((ashape[1] % bshape[1] == 0), "Number of q_heads must be divisible by kv_heads!");
    TT_FATAL((ashape[2] == bshape[0]), "Num of users must match!");
    TT_FATAL((bshape[0] == 32), "Only batch 32 is supported for group attention matmul!");

    const auto num_cores_used =
        std::max(ashape[1], tt::constants::TILE_HEIGHT);  // Need at least 32 cores for mcasting KV heads
    TT_FATAL(
        (num_cores_used <=
         operation_attributes.compute_with_storage_grid_size.x * operation_attributes.compute_with_storage_grid_size.y),
        "Compute grid size is too small for group attention matmul! For now, we require at most 1 q_heads per core.");

    // Any sharded memory configs must be HEIGHT_SHARDED and have the same orientation
    ShardOrientation shard_orientation =
        operation_attributes.row_major ? ShardOrientation::ROW_MAJOR : ShardOrientation::COL_MAJOR;
    if (input_tensor_a.is_sharded()) {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Input tensor A memory layout must be HEIGHT_SHARDED but got {}",
            input_tensor_a.memory_config().memory_layout());
        TT_FATAL(
            input_tensor_a.shard_spec().value().orientation == shard_orientation,
            "Any sharded memory configs must have the same shard orientation as one another!");
        TT_FATAL(
            input_tensor_a.shard_spec().value().num_cores() == ashape[1],
            "Q heads must be sharded on number of q heads!");
        auto shard_shape = input_tensor_a.shard_spec().value().shape;
        TT_FATAL(
            shard_shape[0] == ashape[2],
            "Input tensor A shard height ({}) must equal batch size ({})",
            shard_shape[0],
            ashape[2]);
        TT_FATAL(
            shard_shape[1] == ashape[3],
            "Input tensor A shard width ({}) must equal head dimension ({})",
            shard_shape[1],
            ashape[3]);
    }
    if (input_tensor_b.is_sharded()) {
        TT_FATAL(
            input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Input tensor B memory layout must be HEIGHT_SHARDED but got {}",
            input_tensor_b.memory_config().memory_layout());
        TT_FATAL(
            input_tensor_b.shard_spec().value().orientation == shard_orientation,
            "Any sharded memory configs must have the same shard orientation as one another!");
        TT_FATAL(input_tensor_b.shard_spec().value().num_cores() == bshape[0], "KV heads must be sharded on batch!");
        auto shard_shape = input_tensor_b.shard_spec().value().shape;
        TT_FATAL(
            shard_shape[0] == bshape[1] * bshape[2],
            "Input tensor B shard height ({}) must equal KV heads * head dimension ({} * {})",
            shard_shape[0],
            bshape[1],
            bshape[2]);
        TT_FATAL(
            shard_shape[1] == bshape[3],
            "Input tensor B shard width ({}) must equal KV length ({})",
            shard_shape[1],
            bshape[3]);
    }
    if (operation_attributes.output_mem_config.is_sharded()) {
        TT_FATAL(
            operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Output memory config layout must be HEIGHT_SHARDED but got {}",
            operation_attributes.output_mem_config.memory_layout());

        // If user passes in output_mem_config with shard_spec, assert that it is the same as the one calculated in
        // GroupAttnMatmulDeviceOperation::create_output_tensors
        if (operation_attributes.output_mem_config.shard_spec().has_value()) {
            const ttnn::Shape output_shape = compute_output_specs(operation_attributes, tensor_args).padded_shape();
            const uint32_t num_cores = output_shape[1];
            CoreRangeSet all_cores = num_cores_to_corerangeset(
                num_cores, operation_attributes.compute_with_storage_grid_size, operation_attributes.row_major);

            auto shard_shape = operation_attributes.output_mem_config.shard_spec().value().shape;
            TT_FATAL(
                operation_attributes.output_mem_config.shard_spec().value().grid == all_cores,
                "Shard spec in output mem config must match shard spec calculated in "
                "GroupAttnMatmulDeviceOperation::create_output_tensors!");
            TT_FATAL(
                operation_attributes.output_mem_config.shard_spec().value().orientation == shard_orientation,
                "Any sharded memory configs must have the same shard orientation as one another!");
            TT_FATAL(
                shard_shape[0] == output_shape[2],
                "Output shard height ({}) must equal batch size ({})",
                shard_shape[0],
                output_shape[2]);
            TT_FATAL(
                shard_shape[1] == output_shape[3],
                "Output shard width ({}) must equal KV length ({})",
                shard_shape[1],
                output_shape[3]);
        }
    }

    bool read_from_kv_cache = false;
    if (operation_attributes.num_tokens.has_value() or operation_attributes.transpose_hw.has_value()) {
        TT_FATAL(
            (operation_attributes.num_tokens.has_value() and operation_attributes.transpose_hw.has_value()),
            "Must provide num_tokens and transpose_hw flag if we are reading from cache for in1!");
        TT_FATAL(operation_attributes.num_tokens.value() % 32 == 0, "Number of tokens must be divisble by 32!");
        read_from_kv_cache = true;
    }

    if (read_from_kv_cache) {
        if (operation_attributes.transpose_hw.value()) {
            TT_FATAL(
                ashape[3] == bshape[3],
                "For pre-attention matmul, dimension K for B is in B.shape[3], so A.shape[3] must match B.shape[3]");  // A.K == B.K
        } else {
            TT_FATAL(
                ashape[3] == operation_attributes.num_tokens,
                "For post-attention matmul, dimension K (A.shape[3]) is the kv_seq_len in this case and must match the "
                "length of the cache we read");  // A.K == B.K
        }
    } else {
        TT_FATAL(
            ashape[3] == bshape[2],
            "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in attn_matmul op");  // A.K == B.K
    }
}

GroupAttnMatmulDeviceOperation::spec_return_value_t GroupAttnMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // input_a: [q_len, q_heads, batch, head_dim]
    // input_b: [batch, kv_heads, head_dim, kv_len]
    // intermediate: [q_heads, batch, batch, kv_len]
    // output: [q_len, q_heads, batch, kv_len]
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& ashape = input_tensor_a.padded_shape();
    const auto& bshape = input_tensor_b.padded_shape();

    uint32_t N = bshape[3];
    if (operation_attributes.transpose_hw.value_or(false)) {
        N = operation_attributes.num_tokens.value();
    }
    Shape output_shape({1, ashape[1], ashape[2], N});

    if (operation_attributes.output_mem_config.is_sharded()) {
        auto output_mem_config = operation_attributes.output_mem_config;
        if (!operation_attributes.output_mem_config.shard_spec().has_value()) {
            const uint32_t num_cores = output_shape[1];
            CoreRangeSet all_cores = num_cores_to_corerangeset(
                num_cores, operation_attributes.compute_with_storage_grid_size, operation_attributes.row_major);

            ShardOrientation shard_orientation =
                operation_attributes.row_major ? ShardOrientation::ROW_MAJOR : ShardOrientation::COL_MAJOR;
            ShardSpec shard_spec = ShardSpec{all_cores, {output_shape[2], output_shape[3]}, shard_orientation};
            output_mem_config = output_mem_config.with_shard_spec(shard_spec);
        }
        return TensorSpec(
            output_shape, TensorLayout(operation_attributes.output_dtype, PageConfig(Layout::TILE), output_mem_config));
    }
    return TensorSpec(
        output_shape,
        TensorLayout(
            operation_attributes.output_dtype, PageConfig(Layout::TILE), operation_attributes.output_mem_config));
}

GroupAttnMatmulDeviceOperation::tensor_return_value_t GroupAttnMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor_a.device());
}

tt::stl::hash::hash_t GroupAttnMatmulDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    TT_ASSERT(
        std::holds_alternative<DeviceStorage>(input_tensor_a.storage()),
        "Unexpected type {}",
        tt::stl::get_active_type_name_in_variant(input_tensor_a.storage()));
    TT_ASSERT(
        std::holds_alternative<DeviceStorage>(input_tensor_b.storage()),
        "Unexpected type {}",
        tt::stl::get_active_type_name_in_variant(input_tensor_b.storage()));

    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return operation::hash_operation<GroupAttnMatmulDeviceOperation>(
        program_factory.index(),
        operation_attributes.transpose_hw,
        operation_attributes.out_subblock_w,
        operation_attributes.compute_with_storage_grid_size.str(),
        operation_attributes.output_mem_config.memory_layout(),
        operation_attributes.output_mem_config.buffer_type(),
        operation_attributes.output_dtype,
        operation_attributes.row_major,
        operation_attributes.compute_kernel_config,  // Affects math_fidelity and fp32_dest_acc_en in ComputeConfig
        input_tensor_a.memory_config().memory_layout(),
        input_tensor_a.memory_config().buffer_type(),
        input_tensor_a.dtype(),
        input_tensor_a.device()->id(),
        input_tensor_b.memory_config().memory_layout(),
        input_tensor_b.memory_config().buffer_type(),
        input_tensor_b.dtype(),
        input_tensor_b.device()->id());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor group_attn_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const CoreCoord& compute_with_storage_grid_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> output_dtype,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<const uint32_t> num_tokens,
    std::optional<const bool> transpose_hw,
    uint32_t out_subblock_w,
    bool row_major,
    std::optional<Tensor> preallocated_output) {
    using OperationType = ttnn::experimental::prim::GroupAttnMatmulDeviceOperation;

    auto operation_attributes = ttnn::experimental::prim::GroupAttnMatmulParams{
        .num_tokens = num_tokens,
        .transpose_hw = transpose_hw,
        .out_subblock_w = out_subblock_w,
        .compute_with_storage_grid_size = compute_with_storage_grid_size,
        .output_mem_config = memory_config.value_or(input_tensor_a.memory_config()),
        .output_dtype = output_dtype.value_or(input_tensor_a.dtype()),
        .row_major = row_major,
        .compute_kernel_config = compute_kernel_config.value_or(ttnn::DeviceComputeKernelConfig{}),
    };

    auto tensor_args = ttnn::experimental::prim::GroupAttnMatmulInputs{
        .input_tensor_a = input_tensor_a,
        .input_tensor_b = input_tensor_b,
        .preallocated_output = std::move(preallocated_output),
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
