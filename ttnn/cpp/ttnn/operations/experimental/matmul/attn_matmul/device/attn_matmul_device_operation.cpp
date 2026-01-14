// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_matmul_device_operation.hpp"
#include "attn_matmul_program_factory.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::matmul::attn_matmul {

AttnMatmulDeviceOperation::program_factory_t AttnMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::AttnMatmulProgramFactory{};
}

void AttnMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void AttnMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
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
    TT_FATAL((bshape[1] == 1), "Number of kv_heads must be 1!");  // TODO: May need to uplift to support falcon-40B
    TT_FATAL((ashape[2] == bshape[0]), "Num of users must match!");

    bool read_from_kv_cache = false;
    if (args.num_tokens.has_value() or args.transpose_hw.has_value()) {
        TT_FATAL(
            (args.num_tokens.has_value() and args.transpose_hw.has_value()),
            "Must provide num_tokens and transpose_hw flag if we are reading from cache for in1!");
        TT_FATAL(args.num_tokens.value() % 32 == 0, "Number of tokens must be divisble by 32!");
        read_from_kv_cache = true;
    }

    if (read_from_kv_cache) {
        if (args.transpose_hw.value()) {
            TT_FATAL(
                ashape[3] == bshape[3],
                "For pre-attention matmul, dimension K for B is in B.shape[3], so A.shape[3] must match B.shape[3]");  // A.K == B.K
        } else {
            TT_FATAL(
                ashape[3] == args.num_tokens,
                "For post-attention matmul, dimension K (A.shape[3]) is the kv_seq_len in this case and must match the "
                "length of the cache we read");  // A.K == B.K
        }
    } else {
        TT_FATAL(
            ashape[3] == bshape[2],
            "Dimension K (A.shape[3]and B.shape[2]) must match for A shape: {} and B shape: {} in attn_matmul op",
            ashape,
            bshape);  // A.K == B.K
    }

    auto device_compute_with_storage_grid_size = input_tensor_a.device()->compute_with_storage_grid_size();
    TT_ASSERT(
        (args.compute_with_storage_grid_size.x <= device_compute_with_storage_grid_size.x &&
         args.compute_with_storage_grid_size.y <= device_compute_with_storage_grid_size.y),
        "Unsupported grid shape");
}

AttnMatmulDeviceOperation::spec_return_value_t AttnMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // input_a: [q_len, q_heads, batch, head_dim]
    // input_b: [batch, kv_heads, head_dim, kv_len]
    // intermediate: [q_heads, batch, batch, kv_len]
    // output: [q_len, q_heads, batch, kv_len]
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& ashape = input_tensor_a.padded_shape();
    const auto& bshape = input_tensor_b.padded_shape();

    uint32_t N = bshape[3];
    if (args.transpose_hw.value_or(false)) {
        N = args.num_tokens.value();
    }
    Shape shape({1, ashape[1], ashape[2], N});
    return TensorSpec(shape, TensorLayout(args.output_dtype, PageConfig(Layout::TILE), args.output_mem_config));
}

AttnMatmulDeviceOperation::tensor_return_value_t AttnMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor_a.device());
}

tt::stl::hash::hash_t AttnMatmulDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    TT_ASSERT(
        std::holds_alternative<DeviceStorage>(tensor_args.input_tensor_a.storage()),
        "Unexpected type {}",
        tt::stl::get_active_type_name_in_variant(tensor_args.input_tensor_a.storage()));
    TT_ASSERT(
        std::holds_alternative<DeviceStorage>(tensor_args.input_tensor_b.storage()),
        "Unexpected type {}",
        tt::stl::get_active_type_name_in_variant(tensor_args.input_tensor_b.storage()));

    auto program_factory = select_program_factory(args, tensor_args);

    return operation::hash_operation<AttnMatmulDeviceOperation>(
        args,
        program_factory.index(),
        args.transpose_hw,
        args.output_mem_config,
        args.output_dtype,
        tensor_args.input_tensor_a.dtype(),
        tensor_args.input_tensor_a.memory_config(),
        tensor_args.input_tensor_b.dtype(),
        tensor_args.input_tensor_b.memory_config());
}

}  // namespace ttnn::operations::experimental::matmul::attn_matmul

namespace ttnn::prim {

ttnn::operations::experimental::matmul::attn_matmul::AttnMatmulDeviceOperation::tensor_return_value_t attn_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const CoreCoord& compute_with_storage_grid_size,
    std::optional<const DataType> output_dtype,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const uint32_t> num_tokens,
    std::optional<const bool> transpose_hw,
    std::optional<Tensor> optional_output_tensor) {
    using OperationType = ttnn::operations::experimental::matmul::attn_matmul::AttnMatmulDeviceOperation;

    auto arch = input_tensor_a.device()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config);

    auto operation_attributes = OperationType::operation_attributes_t{
        num_tokens,
        transpose_hw,
        compute_with_storage_grid_size,
        memory_config.value_or(input_tensor_a.memory_config()),
        output_dtype.value_or(input_tensor_a.dtype()),
        kernel_config_val};

    auto tensor_args = OperationType::tensor_args_t{input_tensor_a, input_tensor_b, std::move(optional_output_tensor)};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
