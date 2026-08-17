// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "moe_fused_swiglu_device_operation.hpp"

#include <initializer_list>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu {
namespace {

constexpr uint32_t TILE = tt::constants::TILE_HEIGHT;

bool is_dram_interleaved(const Tensor& tensor) {
    const auto& memory_config = tensor.memory_config();
    return memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM &&
           memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
}

bool is_dram_nd_sharded(const Tensor& tensor) {
    const auto& memory_config = tensor.memory_config();
    return memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM && memory_config.created_with_nd_shard_spec();
}

void validate_device_tensor(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "moe_fused_swiglu: {} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "moe_fused_swiglu: {} must have an allocated device buffer", name);
}

void validate_aux_shape(const Tensor& tensor, const char* name) {
    const auto& shape = tensor.logical_shape();
    TT_FATAL(
        shape.rank() == 1 || (shape.rank() == 2 && shape[0] == 1),
        "moe_fused_swiglu: {} must be 1D or shape (1, N), got {}",
        name,
        shape);
    TT_FATAL(shape[-1] > 0, "moe_fused_swiglu: {} must not be empty", name);
}

}  // namespace

void MoeFusedSwiGluDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_arguments, const tensor_args_t& tensor_arguments) {
    for (const auto& [name, tensor] : std::initializer_list<std::pair<const char*, const Tensor&>>{
             {"activations", tensor_arguments.activations},
             {"w_gate", tensor_arguments.w_gate},
             {"w_up", tensor_arguments.w_up},
             {"w_down", tensor_arguments.w_down},
             {"counts", tensor_arguments.counts},
             {"global_expert_idx_table", tensor_arguments.global_expert_idx_table}}) {
        validate_device_tensor(tensor, name);
        TT_FATAL(
            tensor.device() == tensor_arguments.activations.device(),
            "moe_fused_swiglu: {} must be on the activations device",
            name);
    }

    const auto& activation_shape = tensor_arguments.activations.logical_shape();
    TT_FATAL(
        activation_shape.rank() == 2 || activation_shape.rank() == 4,
        "moe_fused_swiglu: activations must have rank 2 or 4, got {}",
        activation_shape.rank());
    if (activation_shape.rank() == 4) {
        TT_FATAL(
            activation_shape[0] == 1 && activation_shape[1] == 1,
            "moe_fused_swiglu: rank-4 activations leading dimensions must be (1, 1), got ({}, {})",
            activation_shape[0],
            activation_shape[1]);
    }
    TT_FATAL(
        activation_shape[-2] % TILE == 0,
        "moe_fused_swiglu: activation capacity ({}) must be tile-aligned",
        activation_shape[-2]);
    const bool activations_are_row_major = tensor_arguments.activations.layout() == tt::tt_metal::Layout::ROW_MAJOR;
    const bool activations_are_tiled = tensor_arguments.activations.layout() == tt::tt_metal::Layout::TILE;
    TT_FATAL(
        (activations_are_row_major && tensor_arguments.activations.dtype() == tt::tt_metal::DataType::BFLOAT16) ||
            (activations_are_tiled && tensor_arguments.activations.dtype() == tt::tt_metal::DataType::BFLOAT8_B),
        "moe_fused_swiglu: activations must be BFLOAT16 ROW_MAJOR or BFLOAT8_B TILE "
        "(got dtype {}, layout {})",
        tensor_arguments.activations.dtype(),
        tensor_arguments.activations.layout());
    TT_FATAL(
        is_dram_interleaved(tensor_arguments.activations), "moe_fused_swiglu: activations must be DRAM interleaved");

    const auto& gate_shape = tensor_arguments.w_gate.logical_shape();
    const auto& up_shape = tensor_arguments.w_up.logical_shape();
    const auto& down_shape = tensor_arguments.w_down.logical_shape();
    for (const auto& [name, weight] : std::initializer_list<std::pair<const char*, const Tensor&>>{
             {"w_gate", tensor_arguments.w_gate},
             {"w_up", tensor_arguments.w_up},
             {"w_down", tensor_arguments.w_down}}) {
        TT_FATAL(weight.logical_shape().rank() == 2, "moe_fused_swiglu: {} must have rank 2", name);
        TT_FATAL(weight.layout() == tt::tt_metal::Layout::TILE, "moe_fused_swiglu: {} must be TILE layout", name);
        TT_FATAL(
            weight.dtype() == tt::tt_metal::DataType::BFLOAT4_B ||
                weight.dtype() == tt::tt_metal::DataType::BFLOAT8_B ||
                weight.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "moe_fused_swiglu: {} dtype must be BFLOAT4_B, BFLOAT8_B, or BFLOAT16 (got {})",
            name,
            weight.dtype());
        TT_FATAL(
            is_dram_interleaved(weight) || is_dram_nd_sharded(weight),
            "moe_fused_swiglu: {} must be DRAM interleaved or DRAM ND-sharded",
            name);
    }
    TT_FATAL(
        tensor_arguments.w_gate.dtype() == tensor_arguments.w_up.dtype() &&
            tensor_arguments.w_gate.dtype() == tensor_arguments.w_down.dtype(),
        "moe_fused_swiglu: all three weights must have the same dtype (gate={}, up={}, down={})",
        tensor_arguments.w_gate.dtype(),
        tensor_arguments.w_up.dtype(),
        tensor_arguments.w_down.dtype());
    TT_FATAL(
        gate_shape == up_shape,
        "moe_fused_swiglu: gate and up shapes must match (got {} and {})",
        gate_shape,
        up_shape);
    TT_FATAL(
        gate_shape[-2] == activation_shape[-1],
        "moe_fused_swiglu: gate K ({}) must equal activation embedding ({})",
        gate_shape[-2],
        activation_shape[-1]);
    TT_FATAL(
        down_shape[-2] == gate_shape[-1],
        "moe_fused_swiglu: down K ({}) must equal gate/up hidden ({})",
        down_shape[-2],
        gate_shape[-1]);
    TT_FATAL(
        down_shape[-1] == activation_shape[-1],
        "moe_fused_swiglu: down N ({}) must equal activation embedding ({})",
        down_shape[-1],
        activation_shape[-1]);
    TT_FATAL(
        activation_shape[-1] % TILE == 0 && gate_shape[-1] % TILE == 0,
        "moe_fused_swiglu: embedding ({}) and hidden ({}) must be tile-aligned",
        activation_shape[-1],
        gate_shape[-1]);

    for (const auto& [name, aux] : std::initializer_list<std::pair<const char*, const Tensor&>>{
             {"counts", tensor_arguments.counts},
             {"global_expert_idx_table", tensor_arguments.global_expert_idx_table}}) {
        TT_FATAL(aux.dtype() == tt::tt_metal::DataType::UINT32, "moe_fused_swiglu: {} must be UINT32", name);
        TT_FATAL(aux.layout() == tt::tt_metal::Layout::ROW_MAJOR, "moe_fused_swiglu: {} must be ROW_MAJOR", name);
        TT_FATAL(is_dram_interleaved(aux), "moe_fused_swiglu: {} must be DRAM interleaved", name);
        validate_aux_shape(aux, name);
    }
    TT_FATAL(
        operation_arguments.local_expert_id < tensor_arguments.global_expert_idx_table.logical_shape()[-1],
        "moe_fused_swiglu: local_expert_id {} is out of range for idx table length {}",
        operation_arguments.local_expert_id,
        tensor_arguments.global_expert_idx_table.logical_shape()[-1]);
    TT_FATAL(operation_arguments.m_tiles > 0, "moe_fused_swiglu: input_m_tiles must be positive");
    TT_FATAL(
        operation_arguments.m_tiles <= tensor_arguments.activations.padded_shape()[-2] / TILE,
        "moe_fused_swiglu: input_m_tiles {} exceeds activation capacity {} tiles",
        operation_arguments.m_tiles,
        tensor_arguments.activations.padded_shape()[-2] / TILE);
    TT_FATAL(
        operation_arguments.grid_x >= operation_arguments.grid_y && operation_arguments.grid_y >= 2,
        "moe_fused_swiglu: core grid must have columns >= rows >= 2, got {}x{}",
        operation_arguments.grid_x,
        operation_arguments.grid_y);
    const auto available_grid = tensor_arguments.activations.device()->compute_with_storage_grid_size();
    TT_FATAL(
        operation_arguments.grid_x <= available_grid.x && operation_arguments.grid_y <= available_grid.y,
        "moe_fused_swiglu: requested grid {}x{} exceeds device grid {}x{}",
        operation_arguments.grid_x,
        operation_arguments.grid_y,
        available_grid.x,
        available_grid.y);

    TT_FATAL(
        operation_arguments.output_dtype == tt::tt_metal::DataType::BFLOAT8_B ||
            operation_arguments.output_dtype == tt::tt_metal::DataType::BFLOAT16,
        "moe_fused_swiglu: output dtype must be BFLOAT8_B or BFLOAT16");
    TT_FATAL(
        operation_arguments.output_memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM &&
            operation_arguments.output_memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "moe_fused_swiglu: output must be DRAM interleaved");
    TT_FATAL(
        operation_arguments.compute_kernel_config.has_value(),
        "moe_fused_swiglu: compute kernel configuration must be resolved before launching the primitive");
    const auto& compute_config = *operation_arguments.compute_kernel_config;
    TT_FATAL(
        !compute_config.fp32_dest_acc_en,
        "moe_fused_swiglu: fp32_dest_acc_en is unsupported because the kernel blocking requires eight DEST tiles");
    TT_FATAL(
        !compute_config.packer_l1_acc,
        "moe_fused_swiglu: packer_l1_acc is managed explicitly inside the fused kernel and must be false");
    TT_FATAL(
        !compute_config.dst_full_sync_en,
        "moe_fused_swiglu: dst_full_sync_en is unsupported by the BF16 row-major tilize path");

    const bool direct_write = tensor_arguments.expert_region_offsets.has_value();
    TT_FATAL(
        !operation_arguments.read_x_at_offset || direct_write,
        "moe_fused_swiglu: read_x_at_offset requires expert_region_offsets");
    if (direct_write) {
        const auto& offsets = *tensor_arguments.expert_region_offsets;
        validate_device_tensor(offsets, "expert_region_offsets");
        TT_FATAL(
            offsets.device() == tensor_arguments.activations.device(),
            "moe_fused_swiglu: expert_region_offsets must use the activations device");
        TT_FATAL(
            offsets.dtype() == tt::tt_metal::DataType::UINT32 && offsets.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
                is_dram_interleaved(offsets),
            "moe_fused_swiglu: expert_region_offsets must be UINT32 ROW_MAJOR DRAM interleaved");
        const auto& offsets_shape = offsets.logical_shape();
        validate_aux_shape(offsets, "expert_region_offsets");
        TT_FATAL(
            offsets_shape[-1] == tensor_arguments.counts.logical_shape()[-1],
            "moe_fused_swiglu: expert_region_offsets length {} must equal counts length {}",
            offsets_shape[-1],
            tensor_arguments.counts.logical_shape()[-1]);
        TT_FATAL(tensor_arguments.optional_output.has_value(), "moe_fused_swiglu: direct-write mode requires output");
    }

    if (tensor_arguments.optional_output.has_value()) {
        const auto& output = *tensor_arguments.optional_output;
        validate_device_tensor(output, "output");
        TT_FATAL(
            output.device() == tensor_arguments.activations.device(),
            "moe_fused_swiglu: output must use the activations device");
        TT_FATAL(output.layout() == tt::tt_metal::Layout::TILE, "moe_fused_swiglu: output must be TILE layout");
        TT_FATAL(
            output.dtype() == operation_arguments.output_dtype,
            "moe_fused_swiglu: output dtype contradicts dtype argument");
        TT_FATAL(
            output.memory_config() == operation_arguments.output_memory_config,
            "moe_fused_swiglu: output memory config contradicts memory_config argument");
        TT_FATAL(
            output.logical_shape().rank() == activation_shape.rank(),
            "moe_fused_swiglu: output rank {} must match activation rank {}",
            output.logical_shape().rank(),
            activation_shape.rank());
        if (activation_shape.rank() == 4) {
            TT_FATAL(
                output.logical_shape()[0] == activation_shape[0] && output.logical_shape()[1] == activation_shape[1],
                "moe_fused_swiglu: output leading dimensions ({}, {}) must match activations ({}, {})",
                output.logical_shape()[0],
                output.logical_shape()[1],
                activation_shape[0],
                activation_shape[1]);
        }
        TT_FATAL(
            output.logical_shape()[-1] == activation_shape[-1],
            "moe_fused_swiglu: output embedding {} must equal activation embedding {}",
            output.logical_shape()[-1],
            activation_shape[-1]);
        TT_FATAL(output.logical_shape()[-2] % TILE == 0, "moe_fused_swiglu: output rows must be tile-aligned");
        if (direct_write) {
            TT_FATAL(
                output.logical_shape()[-2] >= activation_shape[-2],
                "moe_fused_swiglu: shared output rows {} must be >= activation rows {}",
                output.logical_shape()[-2],
                activation_shape[-2]);
        } else {
            TT_FATAL(
                output.logical_shape() == activation_shape,
                "moe_fused_swiglu: output shape {} must equal activation shape {} outside direct-write mode",
                output.logical_shape(),
                activation_shape);
        }
        for (const auto& [name, tensor] : std::initializer_list<std::pair<const char*, const Tensor&>>{
                 {"activations", tensor_arguments.activations},
                 {"w_gate", tensor_arguments.w_gate},
                 {"w_up", tensor_arguments.w_up},
                 {"w_down", tensor_arguments.w_down},
                 {"counts", tensor_arguments.counts},
                 {"global_expert_idx_table", tensor_arguments.global_expert_idx_table}}) {
            TT_FATAL(
                output.buffer()->address() != tensor.buffer()->address(),
                "moe_fused_swiglu: output must not alias {}; device readers can overlap output writeback",
                name);
        }
        if (tensor_arguments.expert_region_offsets.has_value()) {
            TT_FATAL(
                output.buffer()->address() != tensor_arguments.expert_region_offsets->buffer()->address(),
                "moe_fused_swiglu: output must not alias expert_region_offsets");
        }
    }
}

void MoeFusedSwiGluDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_arguments, const tensor_args_t& tensor_arguments) {
    // Tensor specs and compile-time attributes participate in the cache key,
    // but buffer addresses do not. Re-run validation so an address-only cache
    // hit cannot introduce an output/activation alias or move one argument to a
    // different device after a valid program has already been cached.
    validate_on_program_cache_miss(operation_arguments, tensor_arguments);
}

MoeFusedSwiGluDeviceOperation::spec_return_value_t MoeFusedSwiGluDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_arguments, const tensor_args_t& tensor_arguments) {
    if (tensor_arguments.optional_output.has_value()) {
        return tensor_arguments.optional_output->tensor_spec();
    }
    return tt::tt_metal::TensorSpec(
        tensor_arguments.activations.logical_shape(),
        tt::tt_metal::TensorLayout(
            operation_arguments.output_dtype,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            operation_arguments.output_memory_config));
}

MoeFusedSwiGluDeviceOperation::tensor_return_value_t MoeFusedSwiGluDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_arguments, const tensor_args_t& tensor_arguments) {
    if (tensor_arguments.optional_output.has_value()) {
        return *tensor_arguments.optional_output;
    }
    return create_device_tensor(
        compute_output_specs(operation_arguments, tensor_arguments), tensor_arguments.activations.device());
}

tt::tt_metal::ProgramDescriptor MoeFusedSwiGluDeviceOperation::create_descriptor(
    const operation_attributes_t& operation_arguments,
    const tensor_args_t& tensor_arguments,
    tensor_return_value_t& output) {
    return create_moe_fused_swiglu_program_descriptor(operation_arguments, tensor_arguments, output);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu

namespace ttnn::prim {

ttnn::Tensor moe_fused_swiglu(
    const ttnn::Tensor& activations,
    const ttnn::Tensor& w_gate,
    const ttnn::Tensor& w_up,
    const ttnn::Tensor& w_down,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    uint32_t local_expert_id,
    uint32_t m_tiles,
    uint32_t grid_x,
    uint32_t grid_y,
    bool read_x_at_offset,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& optional_output,
    const std::optional<ttnn::Tensor>& expert_region_offsets) {
    using OperationType = operations::experimental::deepseek_prefill::moe_fused_swiglu::MoeFusedSwiGluDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .local_expert_id = local_expert_id,
            .m_tiles = m_tiles,
            .grid_x = grid_x,
            .grid_y = grid_y,
            .read_x_at_offset = read_x_at_offset,
            .output_dtype = output_dtype,
            .output_memory_config = output_memory_config,
            .compute_kernel_config = compute_kernel_config},
        OperationType::tensor_args_t{
            .activations = activations,
            .w_gate = w_gate,
            .w_up = w_up,
            .w_down = w_down,
            .counts = counts,
            .global_expert_idx_table = global_expert_idx_table,
            .optional_output = optional_output,
            .expert_region_offsets = expert_region_offsets});
}

}  // namespace ttnn::prim
