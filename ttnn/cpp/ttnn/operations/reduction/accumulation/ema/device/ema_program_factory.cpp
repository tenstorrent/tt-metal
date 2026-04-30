// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ema_device_operation.hpp"

#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <bit>

namespace ttnn::prim {

using namespace tt::tt_metal;

constexpr auto ema_buffer_depth = 2;

tt::tt_metal::ProgramDescriptor EmaDeviceOperation::EmaProgramFactory::create_descriptor(
    const EmaParams& operation_attributes, const EmaInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    auto& output = tensor_return_value;
    IDevice* device = input.device();

    // Grid sizing
    // -----------
    // If empty grid size, use all cores
    auto grid_size = operation_attributes.grid_size;
    if ((grid_size.x == 0) && (grid_size.y == 0)) {
        grid_size = device->compute_with_storage_grid_size();
    }
    auto num_cores_available = grid_size.x * grid_size.y;

    // Compute total_tiles to determine core split
    auto input_shape = input.padded_shape();
    auto num_batches = input_shape[1];
    auto num_channels = input_shape[2];
    auto num_samples_per_channel = input_shape[3];

    auto num_channel_tiles = num_channels / input.tensor_spec().tile().get_height();
    auto tiles_per_channel = num_samples_per_channel / input.tensor_spec().tile().get_width();

    auto total_batch_channel_tiles = num_batches * num_channel_tiles;

    // We pick the maximum number of cores (from the available) that divides total_tiles equally
    auto [num_cores, total_batch_channel_tiles_per_core] = get_max_cores_divisible_by_tiles_per_core_tiles(
        total_batch_channel_tiles, num_cores_available, /*request_even=*/false);

    // We now have the number of cores to use, compute per core parameters
    auto all_cores = CoreRangeSet(grid_to_cores(num_cores, grid_size.x, grid_size.y, false));

    log_debug(
        tt::LogOp,
        "EmaProgramFactory: grid_size=({}, {}), num_cores={}, total_batch_channel_tiles={}",
        grid_size.y,
        grid_size.x,
        num_cores,
        total_batch_channel_tiles);

    auto total_tiles_per_core = total_batch_channel_tiles_per_core * tiles_per_channel;

    // Precompute the alpha and beta bits
    // Used by the EMA SFPU instructions
    // ----------------------------------
    auto alpha_bits = std::bit_cast<uint32_t>(operation_attributes.alpha);
    auto beta_bits = std::bit_cast<uint32_t>(1.0f - operation_attributes.alpha);

    // Create program descriptor
    // -------------------------
    ProgramDescriptor desc;

    // Circular buffer config
    // ----------------------
    constexpr auto src_cb_index = tt::CBIndex::c_0;
    constexpr auto dst_cb_index = tt::CBIndex::c_1;
    constexpr auto prev_cb_index = tt::CBIndex::c_2;

    auto src_data_format = datatype_to_dataformat_converter(input.dtype());
    auto dst_data_format = datatype_to_dataformat_converter(output.dtype());

    auto src_tile_size = input.tensor_spec().tile().get_tile_size(src_data_format);
    auto dst_tile_size = output.tensor_spec().tile().get_tile_size(dst_data_format);

    auto src_cb_size = src_tile_size * ema_buffer_depth;
    auto dst_cb_size = dst_tile_size * ema_buffer_depth;
    auto prev_cb_size = src_tile_size;

    desc.cbs.push_back(CBDescriptor{
        .total_size = src_cb_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src_cb_index),
            .data_format = src_data_format,
            .page_size = src_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = dst_cb_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(dst_cb_index),
            .data_format = dst_data_format,
            .page_size = dst_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = prev_cb_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(prev_cb_index),
            .data_format = src_data_format,
            .page_size = src_tile_size,
        }}},
    });

    // Compile time args for the kernels
    // ---------------------------------
    std::vector<uint32_t> reader_compile_args = {total_tiles_per_core};
    TensorAccessorArgs(input.buffer()).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args = {total_tiles_per_core};
    TensorAccessorArgs(output.buffer()).append_to(writer_compile_args);

    std::vector<uint32_t> compute_compile_args = {
        total_batch_channel_tiles_per_core,
        tiles_per_channel,
        alpha_bits,
        beta_bits,
    };

    // Create kernel descriptors
    // -------------------------
    tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/dataflow/ema_reader.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_args);
    reader_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = reader_noc,
    };

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/dataflow/ema_writer.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_args);
    writer_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = writer_noc,
    };

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/reduction/accumulation/ema/kernels/compute/ema_compute.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_compile_args);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    // Set runtime args
    // ---------------
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    uint32_t src_start_tile = 0;
    uint32_t dst_start_tile = 0;
    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            reader_desc.emplace_runtime_args(core, {src_buffer, src_start_tile});
            writer_desc.emplace_runtime_args(core, {dst_buffer, dst_start_tile});
            src_start_tile += total_tiles_per_core;
            dst_start_tile += total_tiles_per_core;
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
