// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/fused_paged_update_and_mla/device/fused_paged_update_and_mla_device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::fused_paged_update_and_mla {

tt::tt_metal::operation::ProgramWithCallbacks FusedPagedUpdateAndMlaDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {

    using namespace tt::constants;
    using namespace tt::tt_metal;

    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    Program program = CreateProgram();
    IDevice* device = input_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    CoreRangeSet all_cores(CoreRange(CoreCoord(0, 0), CoreCoord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1)));

    // 1. L1 Ring Buffer for zero-latency KV access
    uint32_t kv_cb_index = tt::CB::c_in1;
    uint32_t kv_tile_size = 2048; // Bfp8_b approx
    uint32_t num_kv_tiles = 64; 
    CircularBufferConfig cb_kv_config = CircularBufferConfig(2 * num_kv_tiles * kv_tile_size, {{kv_cb_index, tt::DataFormat::Bfp8_b}})
        .set_page_size(kv_cb_index, kv_tile_size);
    auto cb_kv = CreateCircularBuffer(program, all_cores, cb_kv_config);

    // 2. Kernel creation
    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/fused_paged_update_and_mla/device/kernels/dataflow/reader_paged_update_mla.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/fused_paged_update_and_mla/device/kernels/dataflow/writer_paged_update_mla.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );

    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/fused_paged_update_and_mla/device/kernels/compute/paged_update_mla_compute.cpp",
        all_cores,
        ComputeConfig{}
    );

    auto override_runtime_args_callback = [
        reader_kernel_id, writer_kernel_id, compute_kernel_id
    ](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        (void)operation;
        (void)program;
        (void)input_tensors;
        (void)output_tensors;
    };

    return {std::move(program), override_runtime_args_callback};
}

} // namespace ttnn::operations::experimental::fused_paged_update_and_mla
