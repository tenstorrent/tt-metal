// SPDX-FileCopyrightText: © 2025 Tenstorrent AI
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/normalization/groupnorm/device/groupnorm_op.hpp"

namespace ttnn::operations::normalization {

// Define buffer depth for each circular buffer
constexpr uint32_t CB_DEPTH = 2;

operation::ProgramWithCallbacks groupnorm_v3(
    const Tensor& a,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    Tensor& output,
    float eps,
    uint32_t num_groups,
    const CoreCoord& core_grid,
    int chunk_size,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    Program program = Program();

    const auto& in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const auto& out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    auto gamma_beta_cb_data_format = tt::DataFormat::Float16_b;

    // TODO: Move this to the validate() function
    if (gamma.has_value() && beta.has_value()) {
        TT_FATAL(gamma.value().dtype() == beta.value().dtype(), "Gamma and beta must have the same dtype");
    }

    if (gamma.has_value()) {
        gamma_beta_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(gamma.value().dtype());
    }
    if (beta.has_value()) {
        gamma_beta_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(beta.value().dtype());
    }

    const auto total_available_cores = core_grid.x * core_grid.y;
    TT_FATAL(
        num_groups <= total_available_cores,
        "Number of groups ({}) must be <= the number of cores ({})",
        num_groups,
        total_available_cores);

    // Get a subset of cores to use
    std::vector<CoreCoord> cores_used_coords = grid_to_cores(num_groups, core_grid.x, core_grid.y);
    CoreRangeSet cores_used = CoreRangeSet(cores_used_coords);

    // Create circular buffers
    // Input CB
    const uint32_t src_cb_index = tt::CBIndex::c_0;
    const uint32_t src_page_size = chunk_size * tt::datum_size(in_data_format);
    const uint32_t src_tile_size = tt::tt_metal::detail::TileSize(in_data_format);
    const uint32_t src_tiles_per_page = src_page_size / src_tile_size;
    const uint32_t src_cb_size = src_page_size * CB_DEPTH;
    const auto src_cb_config = tt::tt_metal::CircularBufferConfig(src_cb_size, {{src_cb_index, in_data_format}})
                                   .set_page_size(src_cb_index, src_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, cores_used, src_cb_config);

    // Output CB
    const uint32_t dst_cb_index = tt::CBIndex::c_1;
    const uint32_t dst_page_size = chunk_size * tt::datum_size(out_data_format);
    const uint32_t dst_tile_size = tt::tt_metal::detail::TileSize(out_data_format);
    const uint32_t dst_tiles_per_page = dst_page_size / dst_tile_size;
    const uint32_t dst_cb_size = dst_page_size * CB_DEPTH;
    const auto dst_cb_config = tt::tt_metal::CircularBufferConfig(dst_cb_size, {{dst_cb_index, out_data_format}})
                                   .set_page_size(dst_cb_index, dst_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, cores_used, dst_cb_config);

    // Sum CB
    const uint32_t sum_cb_index = tt::CBIndex::c_2;
    const uint32_t sum_cb_size = tt::tt_metal::detail::TileSize(in_data_format);
    const auto sum_cb_config = tt::tt_metal::CircularBufferConfig(sum_cb_size, {{sum_cb_index, in_data_format}})
                                   .set_page_size(sum_cb_index, sum_cb_size);
    tt::tt_metal::CreateCircularBuffer(program, cores_used, sum_cb_config);

    // Mean CB
    const uint32_t mean_cb_index = tt::CBIndex::c_3;
    const uint32_t mean_cb_size = tt::tt_metal::detail::TileSize(in_data_format);
    const auto mean_cb_config = tt::tt_metal::CircularBufferConfig(mean_cb_size, {{mean_cb_index, in_data_format}})
                                    .set_page_size(mean_cb_index, mean_cb_size);
    tt::tt_metal::CreateCircularBuffer(program, cores_used, mean_cb_config);

    // sum of (x - E[x])^2 CB
    const uint32_t varsum_cb_index = tt::CBIndex::c_4;
    const uint32_t varsum_cb_size = tt::tt_metal::detail::TileSize(in_data_format);
    const auto varsum_cb_config =
        tt::tt_metal::CircularBufferConfig(varsum_cb_size, {{varsum_cb_index, in_data_format}})
            .set_page_size(varsum_cb_index, varsum_cb_size);
    tt::tt_metal::CreateCircularBuffer(program, cores_used, varsum_cb_config);

    // Variance CB
    const uint32_t variance_cb_index = tt::CBIndex::c_5;
    const uint32_t variance_cb_size = tt::tt_metal::detail::TileSize(in_data_format);
    const auto variance_cb_config =
        tt::tt_metal::CircularBufferConfig(variance_cb_size, {{variance_cb_index, in_data_format}})
            .set_page_size(variance_cb_index, variance_cb_size);
    tt::tt_metal::CreateCircularBuffer(program, cores_used, variance_cb_config);

    // Sum scaler CB
    const uint32_t sum_scaler_cb_index = tt::CBIndex::c_6;
    const uint32_t sum_scaler_cb_size = tt::tt_metal::detail::TileSize(in_data_format);
    const auto sum_scaler_cb_config =
        tt::tt_metal::CircularBufferConfig(sum_scaler_cb_size, {{sum_scaler_cb_index, in_data_format}})
            .set_page_size(sum_scaler_cb_index, sum_scaler_cb_size);
    tt::tt_metal::CreateCircularBuffer(program, cores_used, sum_scaler_cb_config);

    // Mean scaler CB
    const uint32_t mean_scaler_cb_index = tt::CBIndex::c_7;
    const uint32_t mean_scaler_cb_size = tt::tt_metal::detail::TileSize(in_data_format);
    const auto mean_scaler_cb_config =
        tt::tt_metal::CircularBufferConfig(mean_scaler_cb_size, {{mean_scaler_cb_index, in_data_format}})
            .set_page_size(mean_scaler_cb_index, mean_scaler_cb_size);
    tt::tt_metal::CreateCircularBuffer(program, cores_used, mean_scaler_cb_config);

    const auto N = a.logical_shape()[0];
    const auto C = a.logical_shape()[1];
    const auto H = a.logical_shape()[2];
    const auto W = a.logical_shape()[3];

    const auto pages_per_batch = C * H * W / chunk_size;
    const auto pages_per_group = pages_per_batch / num_groups;

    const auto& device = a.device();

    bool src_is_dram = a.memory_config().buffer_type() == BufferType::DRAM;
    bool dst_is_dram = output.memory_config().buffer_type() == BufferType::DRAM;

    // TODO: Maybe make half of the groups use NOC1 to read? Since there is no write traffic when READs are happening
    auto reader_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_groupnorm_v3.cpp",
        cores_used,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(device->arch()),
            .compile_args =
                {
                    src_is_dram,
                    src_tiles_per_page,
                    src_page_size,
                    pages_per_group,
                    pages_per_batch,
                    N  // num_batches
                },
            .defines = {},
        });

    for (uint32_t i = 0; i < cores_used_coords.size(); ++i) {
        CoreCoord core = cores_used_coords.at(i);
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core, {a.buffer()->address(), i * pages_per_group});
    }

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    auto compute_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_v3.cpp",
        cores_used,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args =
                {src_tiles_per_page,
                 pages_per_group,
                 N,  // num_batches
                 dst_tiles_per_page},
            .defines = {}});

    auto writer_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/writer_groupnorm_v3.cpp",
        cores_used,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::detail::GetPreferredNOCForDRAMWrite(device->arch()),
            .compile_args =
                {
                    dst_is_dram,
                    dst_tiles_per_page,
                    dst_page_size,
                    pages_per_group,
                    pages_per_batch,
                    N,  // num_batches
                },
            .defines = {}});

    for (uint32_t i = 0; i < cores_used_coords.size(); ++i) {
        CoreCoord core = cores_used_coords.at(i);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core, {output.buffer()->address(), i * pages_per_group});
    }

    auto override_runtime_args_callback = [cores_used_coords, reader_kernel, writer_kernel, pages_per_group](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_address = input_tensors.at(0).buffer()->address();
        auto dst_address = output_tensors.at(0).buffer()->address();
        for (uint32_t i = 0; i < cores_used_coords.size(); ++i) {
            CoreCoord core = cores_used_coords.at(i);

            auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel, core);
            reader_runtime_args[0] = src_address;
            reader_runtime_args[1] = i * pages_per_group;

            auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel, core);
            writer_runtime_args[0] = dst_address;
            writer_runtime_args[1] = i * pages_per_group;
        }
    };
    return {std::move(program), override_runtime_args_callback};
}
}  // namespace ttnn::operations::normalization
