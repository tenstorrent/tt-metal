// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include <tt-metalium/circular_buffer_config.hpp>
#include "ema_op.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::operations::reduction::accumulation {

constexpr auto ema_buffer_depth = 2;

tt::tt_metal::operation::ProgramWithCallbacks ema_multi_core(
    const Tensor& a,
    Tensor& output,
    float alpha,
    CoreCoord grid_size,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    // Args to size the grid
    // If empty grid size, use all cores
    if ((grid_size.x == 0) && (grid_size.y == 0)) {
        grid_size = a.device()->compute_with_storage_grid_size();
    }
    auto num_cores = grid_size.x * grid_size.y;
    auto core_grid = CoreRange(CoreCoord(0, 0), CoreCoord(grid_size.x - 1, grid_size.y - 1));

    // Args based on the tensor shape
    auto a_shape = a.padded_shape();
    auto batch = a_shape[0];
    auto channel = a_shape[1];
    auto pages = a_shape[2];

    auto total_pages = batch * channel * pages;
    auto pages_per_core = total_pages / num_cores;

    auto batches_per_core = batch * channel / num_cores;

    auto program = Program();

    // Circular buffer config
    // ----------------------
    auto src_cb_index = tt::CBIndex::c_0;
    auto dst_cb_index = tt::CBIndex::c_1;

    auto a_data_format = datatype_to_dataformat_converter(a.dtype());
    auto dst_data_format = datatype_to_dataformat_converter(output.dtype());

    auto src_cb_size = a.buffer()->aligned_page_size() * ema_buffer_depth;
    auto dst_cb_size = output.buffer()->aligned_page_size() * ema_buffer_depth;

    auto src_cb_cfg = tt::tt_metal::CircularBufferConfig(src_cb_size, {{src_cb_index, a_data_format}})
                          .set_page_size(src_cb_index, a.buffer()->aligned_page_size());
    tt::tt_metal::CreateCircularBuffer(program, core_grid, src_cb_cfg);

    auto dst_cb_cfg = tt::tt_metal::CircularBufferConfig(dst_cb_size, {{dst_cb_index, dst_data_format}})
                          .set_page_size(dst_cb_index, output.buffer()->aligned_page_size());
    tt::tt_metal::CreateCircularBuffer(program, core_grid, dst_cb_cfg);

    // Compile time args for the kernels
    // ---------------------------------
    std::vector<uint32_t> reader_compile_args = {pages_per_core, a.buffer()->aligned_page_size()};
    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args = {
        pages_per_core,
        output.buffer()->aligned_page_size(),
    };
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_args);

    // Create kernels
    // --------------
    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/dataflow/ema_reader.cpp",
        core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_compile_args});

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/dataflow/ema_writer.cpp",
        core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = writer_compile_args});

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), compute_kernel_config);
    CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/compute/ema_compute.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = {batches_per_core, pages}});

    auto all_cores = grid_to_cores(CoreCoord(0, 0), CoreCoord(grid_size.x - 1, grid_size.y - 1), false);
    // Runtime args
    std::vector<uint32_t> reader_runtime_args = {
        a.buffer()->address(),
        0,  // Placeholder for src_start_page
    };
    std::vector<uint32_t> writer_runtime_args = {
        output.buffer()->address(),
        0,  // Placeholder for dst_start_page
    };

    uint32_t src_start_page = 0;
    uint32_t dst_start_page = 0;
    for (const auto& core : all_cores) {
        reader_runtime_args[1] = src_start_page;
        writer_runtime_args[1] = dst_start_page;
        src_start_page += pages_per_core;
        dst_start_page += pages_per_core;

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
    }

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, all_cores](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer()->address();
        auto dst_buffer = output_tensors.at(0).buffer()->address();

        for (const auto& core : all_cores) {
            auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            reader_runtime_args[0] = src_buffer;
            auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            writer_runtime_args[0] = dst_buffer;
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::reduction::accumulation
