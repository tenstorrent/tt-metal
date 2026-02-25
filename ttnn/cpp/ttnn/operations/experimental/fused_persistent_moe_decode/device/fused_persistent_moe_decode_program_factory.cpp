// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/fused_persistent_moe_decode/device/fused_persistent_moe_decode_device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::fused_persistent_moe_decode {

ExecuteFusedPersistentMoeDecodeDeviceOperation::SingleCore::cached_program_t 
ExecuteFusedPersistentMoeDecodeDeviceOperation::SingleCore::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {

    using namespace tt::constants;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& topk_expert_indices = tensor_args.topk_expert_indices;
    const auto& topk_expert_weights = tensor_args.topk_expert_weights;
    const auto& w1_experts = tensor_args.w1_experts;
    const auto& w3_experts = tensor_args.w3_experts;
    const auto& w2_experts = tensor_args.w2_experts;
    auto& output_tensor = tensor_return_value;

    Program program = CreateProgram();
    IDevice* device = input_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    CoreRangeSet all_cores(CoreRange(CoreCoord(0, 0), CoreCoord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1)));

    uint32_t in0_tile_size = input_tensor.buffer()->page_size();
    uint32_t in0_num_tiles = input_tensor.buffer()->num_pages();
    uint32_t out0_tile_size = output_tensor.buffer()->page_size();

    uint32_t cb_in0_index = tt::CB::c_in0;
    CircularBufferConfig cb_in0_config = CircularBufferConfig(2 * in0_tile_size, {{cb_in0_index, tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype())}})
        .set_page_size(cb_in0_index, in0_tile_size);
    auto cb_in0 = CreateCircularBuffer(program, all_cores, cb_in0_config);

    uint32_t cb_out0_index = tt::CB::c_out0;
    CircularBufferConfig cb_out0_config = CircularBufferConfig(2 * out0_tile_size, {{cb_out0_index, tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype())}})
        .set_page_size(cb_out0_index, out0_tile_size);
    auto cb_out0 = CreateCircularBuffer(program, all_cores, cb_out0_config);

    // Double buffered L1 for weights and indices
    uint32_t num_weight_tiles = 256; 
    
    // w1_experts (c_in1)
    uint32_t cb_w1_index = tt::CB::c_in1;
    uint32_t w1_tile_size = w1_experts.buffer()->page_size();
    CircularBufferConfig cb_w1_config = CircularBufferConfig(2 * num_weight_tiles * w1_tile_size, {{cb_w1_index, tt::tt_metal::datatype_to_dataformat_converter(w1_experts.dtype())}})
        .set_page_size(cb_w1_index, w1_tile_size);
    auto cb_w1 = CreateCircularBuffer(program, all_cores, cb_w1_config);

    // w3_experts (c_in2)
    uint32_t cb_w3_index = tt::CB::c_in2;
    uint32_t w3_tile_size = w3_experts.buffer()->page_size();
    CircularBufferConfig cb_w3_config = CircularBufferConfig(2 * num_weight_tiles * w3_tile_size, {{cb_w3_index, tt::tt_metal::datatype_to_dataformat_converter(w3_experts.dtype())}})
        .set_page_size(cb_w3_index, w3_tile_size);
    auto cb_w3 = CreateCircularBuffer(program, all_cores, cb_w3_config);

    // w2_experts (c_in3)
    uint32_t cb_w2_index = tt::CB::c_in3;
    uint32_t w2_tile_size = w2_experts.buffer()->page_size();
    CircularBufferConfig cb_w2_config = CircularBufferConfig(2 * num_weight_tiles * w2_tile_size, {{cb_w2_index, tt::tt_metal::datatype_to_dataformat_converter(w2_experts.dtype())}})
        .set_page_size(cb_w2_index, w2_tile_size);
    auto cb_w2 = CreateCircularBuffer(program, all_cores, cb_w2_config);

    // topk_expert_indices (c_in4)
    uint32_t cb_idx_index = tt::CB::c_in4;
    uint32_t idx_tile_size = topk_expert_indices.buffer()->page_size();
    CircularBufferConfig cb_idx_config = CircularBufferConfig(2 * idx_tile_size, {{cb_idx_index, tt::tt_metal::datatype_to_dataformat_converter(topk_expert_indices.dtype())}})
        .set_page_size(cb_idx_index, idx_tile_size);
    auto cb_idx = CreateCircularBuffer(program, all_cores, cb_idx_config);

    // topk_expert_weights (c_in5)
    uint32_t cb_wt_index = tt::CB::c_in5;
    uint32_t wt_tile_size = topk_expert_weights.buffer()->page_size();
    CircularBufferConfig cb_wt_config = CircularBufferConfig(2 * wt_tile_size, {{cb_wt_index, tt::tt_metal::datatype_to_dataformat_converter(topk_expert_weights.dtype())}})
        .set_page_size(cb_wt_index, wt_tile_size);
    auto cb_wt = CreateCircularBuffer(program, all_cores, cb_wt_config);

    // Intermediate CBs for math
    uint32_t cb_interm0_index = tt::CB::c_intermed0;
    CircularBufferConfig cb_interm0_config = CircularBufferConfig(2 * out0_tile_size, {{cb_interm0_index, tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype())}})
        .set_page_size(cb_interm0_index, out0_tile_size);
    auto cb_interm0 = CreateCircularBuffer(program, all_cores, cb_interm0_config);

    uint32_t cb_interm1_index = tt::CB::c_intermed1;
    CircularBufferConfig cb_interm1_config = CircularBufferConfig(2 * out0_tile_size, {{cb_interm1_index, tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype())}})
        .set_page_size(cb_interm1_index, out0_tile_size);
    auto cb_interm1 = CreateCircularBuffer(program, all_cores, cb_interm1_config);

    // Kernel that stays alive and fetches NOC asynchronously
    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/fused_persistent_moe_decode/device/kernels/dataflow/reader_persistent_moe.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/fused_persistent_moe_decode/device/kernels/dataflow/writer_persistent_moe.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );

    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/fused_persistent_moe_decode/device/kernels/compute/persistent_moe_compute.cpp",
        all_cores,
        ComputeConfig{}
    );

    uint32_t input_addr = input_tensor.buffer()->address();
    uint32_t topk_indices_addr = topk_expert_indices.buffer()->address();
    uint32_t topk_weights_addr = topk_expert_weights.buffer()->address();
    uint32_t w1_addr = w1_experts.buffer()->address();
    uint32_t w3_addr = w3_experts.buffer()->address();
    uint32_t w2_addr = w2_experts.buffer()->address();
    SetRuntimeArgs(program, reader_kernel_id, all_cores, {input_addr, topk_indices_addr, topk_weights_addr, w1_addr, w3_addr, w2_addr, in0_num_tiles});

    uint32_t output_addr = output_tensor.buffer()->address();
    SetRuntimeArgs(program, writer_kernel_id, all_cores, {output_addr, in0_num_tiles});
    
    SetRuntimeArgs(program, compute_kernel_id, all_cores, {in0_num_tiles});

    (void)cb_in0;
    (void)cb_out0;
    (void)cb_w1;
    (void)cb_w3;
    (void)cb_w2;
    (void)cb_idx;
    (void)cb_wt;
    (void)cb_interm0;
    (void)cb_interm1;

    return {std::move(program), {.reader_kernel_id = reader_kernel_id, .writer_kernel_id = writer_kernel_id, .compute_kernel_id = compute_kernel_id}};
}

void ExecuteFusedPersistentMoeDecodeDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    
    auto& program = cached_program.program;
    auto reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto compute_kernel_id = cached_program.shared_variables.compute_kernel_id;
    
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& topk_expert_indices = tensor_args.topk_expert_indices;
    const auto& topk_expert_weights = tensor_args.topk_expert_weights;
    const auto& w1_experts = tensor_args.w1_experts;
    const auto& w3_experts = tensor_args.w3_experts;
    const auto& w2_experts = tensor_args.w2_experts;
    auto& output_tensor = tensor_return_value;

    uint32_t input_addr = input_tensor.buffer()->address();
    uint32_t topk_indices_addr = topk_expert_indices.buffer()->address();
    uint32_t topk_weights_addr = topk_expert_weights.buffer()->address();
    uint32_t w1_addr = w1_experts.buffer()->address();
    uint32_t w3_addr = w3_experts.buffer()->address();
    uint32_t w2_addr = w2_experts.buffer()->address();
    uint32_t output_addr = output_tensor.buffer()->address();
    uint32_t in0_num_tiles = input_tensor.buffer()->num_pages();
    
    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    CoreRangeSet all_cores(CoreRange(CoreCoord(0, 0), CoreCoord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1)));
    
    SetRuntimeArgs(program, reader_kernel_id, all_cores, {input_addr, topk_indices_addr, topk_weights_addr, w1_addr, w3_addr, w2_addr, in0_num_tiles});
    SetRuntimeArgs(program, writer_kernel_id, all_cores, {output_addr, in0_num_tiles});
    SetRuntimeArgs(program, compute_kernel_id, all_cores, {in0_num_tiles});
}

} // namespace ttnn::operations::experimental::fused_persistent_moe_decode
