// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/operations/pool/rotate/device/rotate_device_operation.hpp>
#include <ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp>
#include <ttnn/operations/pool/pool_utils.hpp>

#include <cmath>
#include <cstdint>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::rotate {

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t NEAREST_BUFFERING_FACTOR = 2;
constexpr uint32_t NUM_TILES_DEST = 8;
constexpr uint32_t MAX_BURST_SIZE = 5;

ProgramDescriptor RotateDeviceOperation::NearestProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input;
    auto& output_tensor = output;

    ProgramDescriptor desc;
    const bool is_sharded = input_tensor.is_sharded();

    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::tt_metal::IDevice* const device = output_tensor.device();

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t input_batch = input_shape[0];
    const uint32_t input_height = input_shape[1];
    const uint32_t input_width = input_shape[2];
    const uint32_t input_channels = input_shape[3];

    // Single source of truth for per-core runtime args (shared with get_dynamic_runtime_args).
    const RotatePerCoreArgs per_core_args =
        compute_rotate_per_core_args(operation_attributes, input_tensor, output_tensor, /*is_bilinear=*/false);

    const uint32_t total_output_sticks = input_batch * input_height * input_width;

    const uint32_t element_size = input_tensor.element_size();
    const uint32_t input_stick_nbytes = input_channels * element_size;

    // Work-core grid and CB-sizing quantities. The per-core runtime-arg layout (core list, num_sticks,
    // start_stick_id) comes from per_core_args above; this block only derives the grid-level values the
    // CBs / kernels are sized against.
    tt::tt_metal::CoreRangeSet all_cores;
    uint32_t num_sticks_per_core_group_1 = 0, num_sticks_per_core_group_2 = 0;
    uint32_t input_nsticks_per_core = 0;
    uint32_t output_nsticks_per_core = 0;
    bool is_width_sharded = false;
    uint32_t shard_width = 0;

    const bool is_nd_sharded = input_tensor.memory_config().nd_shard_spec().has_value();

    if (is_sharded && !is_nd_sharded) {
        const auto input_shard_spec = input_tensor.shard_spec().value();
        all_cores = input_shard_spec.grid;
        input_nsticks_per_core = input_shard_spec.shape[0];
        output_nsticks_per_core = output_tensor.shard_spec().value().shape[0];
        is_width_sharded =
            input_tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
        TT_FATAL(!is_width_sharded, "Width sharding is not supported for rotate operation");
        shard_width = input_shard_spec.shape[1];
    } else if (is_nd_sharded) {
        const auto& nd_shard_spec = input_tensor.memory_config().nd_shard_spec().value();
        all_cores = nd_shard_spec.grid;
        const auto& shard_shape = nd_shard_spec.shard_shape;
        input_nsticks_per_core = shard_shape[-3] * shard_shape[-2];
        output_nsticks_per_core = input_nsticks_per_core;
        shard_width = shard_shape[-1];
    } else {
        const auto compute_grid_size = device->compute_with_storage_grid_size();
        auto [num_cores_used, all_cores_range, core_group_1_range, core_group_2_range, num_sticks_1, num_sticks_2] =
            tt::tt_metal::split_work_to_cores(compute_grid_size, total_output_sticks);
        all_cores = all_cores_range;
        num_sticks_per_core_group_1 = num_sticks_1;
        num_sticks_per_core_group_2 = num_sticks_2;
    }

    const bool any_sharded = is_sharded || is_nd_sharded;
    const uint32_t effective_channels = any_sharded ? shard_width : input_channels;
    const uint32_t aligned_input_stick_nbytes = any_sharded ? effective_channels * input_tensor.element_size()
                                                            : pool::get_aligned_stick_size(input_shape, input_tensor);
    const uint32_t aligned_output_stick_nbytes = any_sharded ? effective_channels * output_tensor.element_size()
                                                             : pool::get_aligned_stick_size(input_shape, output_tensor);

    const uint32_t available_l1 = NUM_TILES_DEST * tt::constants::TILE_HW * element_size;
    const uint32_t l1_for_cb = available_l1 / NEAREST_BUFFERING_FACTOR;
    const uint32_t max_cb_pages_from_l1 = l1_for_cb / aligned_input_stick_nbytes;

    const uint32_t max_sticks_per_core =
        any_sharded ? input_nsticks_per_core : std::max(num_sticks_per_core_group_1, num_sticks_per_core_group_2);
    uint32_t num_cb_pages = std::min(max_sticks_per_core, max_cb_pages_from_l1);
    TT_FATAL(
        num_cb_pages > 0,
        "Not enough L1 for even a single CB page: aligned_input_stick_nbytes={} exceeds l1_for_cb={}",
        aligned_input_stick_nbytes,
        l1_for_cb);
    const uint32_t burst_size = num_cb_pages < MAX_BURST_SIZE ? num_cb_pages : MAX_BURST_SIZE;
    // CB total size must be an even multiple of burst_size (required by cb_push_back/cb_pop_front API)
    num_cb_pages = round_down(num_cb_pages, burst_size);

    uint32_t next_cb_index = tt::CBIndex::c_0;
    const uint32_t output_cb_page_size = aligned_input_stick_nbytes;

    const uint32_t fill_cb_index = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(fill_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_cb_page_size,
        }}},
    });

    uint32_t input_cb_index = 0;
    if (any_sharded) {
        input_cb_index = next_cb_index++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = input_nsticks_per_core * aligned_input_stick_nbytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(input_cb_index),
                .data_format = input_cb_data_format,
                .page_size = aligned_input_stick_nbytes,
            }}},
            .buffer = input_tensor.buffer(),
        });
    }

    const uint32_t output_cb_index = next_cb_index++;
    const uint32_t output_cb_num_pages =
        any_sharded ? output_nsticks_per_core : num_cb_pages * NEAREST_BUFFERING_FACTOR;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_num_pages * output_cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_cb_page_size,
        }}},
        .buffer = any_sharded ? output_tensor.buffer() : nullptr,
    });

    const bool fill_is_zero = (per_core_args.fill_value_bits == 0);

    const uint32_t effective_stick_nbytes = any_sharded ? effective_channels * element_size : input_stick_nbytes;

    std::vector<uint32_t> reader_compile_time_args = {
        output_cb_index,
        aligned_input_stick_nbytes,
        input_batch,
        input_height,
        input_width,
        effective_channels,
        num_cb_pages,
        fill_cb_index,
        effective_stick_nbytes,
        static_cast<uint32_t>(fill_is_zero),
        burst_size,
    };

    auto* input_buffer = input_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "Input tensor must be allocated on device for rotate operation");
    tt::tt_metal::TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        output_cb_index,
        aligned_output_stick_nbytes,
        num_cb_pages,
        burst_size,
    };

    auto* output_buffer = output_tensor.buffer();
    TT_FATAL(output_buffer != nullptr, "Output tensor must be allocated on device for rotate operation");
    tt::tt_metal::TensorAccessorArgs(*output_buffer).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/pool/rotate/device/kernels/dataflow/"
        "reader_rotate_nearest_interleaved.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/pool/rotate/device/kernels/dataflow/"
        "writer_rotate_nearest_interleaved.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Emit per-core runtime args from the shared layout. The input/output buffer base addresses are
    // bound as Buffer* (patchable on a cache hit); the angle-derived scalars are re-applied on every
    // hit by get_dynamic_runtime_args.
    for (uint32_t i = 0; i < per_core_args.cores.size(); i++) {
        const CoreCoord& core = per_core_args.cores[i];
        const uint32_t num_sticks = per_core_args.num_sticks[i];
        const uint32_t start_stick_id = per_core_args.start_stick_id[i];

        reader_desc.emplace_runtime_args(
            core,
            {input_tensor.buffer(),
             num_sticks,
             start_stick_id,
             per_core_args.cos_angle_q16,
             per_core_args.sin_angle_q16,
             per_core_args.center_x_q16,
             per_core_args.center_y_q16,
             per_core_args.fill_value_bits});

        writer_desc.emplace_runtime_args(core, {output_tensor.buffer(), num_sticks, start_stick_id});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::operations::rotate
