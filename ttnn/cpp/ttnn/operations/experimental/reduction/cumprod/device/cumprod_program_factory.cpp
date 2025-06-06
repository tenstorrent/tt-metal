// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cumprod_device_operation.hpp"
#include "tt-metalium/base_types.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/util.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::reduction {

uint32_t CumprodDeviceOperation::MultiCoreCumprodProgramFactory::calc_input_tile_offset(
    const Shape& input_shape, const int32_t& dim) {
    uint32_t input_tile_offset{1};
    for (int32_t i = dim + 1; i < input_shape.rank() - 2; ++i) {
        input_tile_offset *= input_shape[i];
    }
    if (input_shape.rank() > 1) {
        input_tile_offset *= (input_shape[-2] / tt::constants::TILE_HEIGHT);
    }
    if (input_shape.rank() > 0) {
        input_tile_offset *= (input_shape[-1] / tt::constants::TILE_WIDTH);
    }

    return input_tile_offset;
}

CumprodDeviceOperation::MultiCoreCumprodProgramFactory::cached_program_t
CumprodDeviceOperation::MultiCoreCumprodProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor{tensor_args.input_tensor};
    auto& output_tensor{tensor_return_value};
    const auto& input_shape{input_tensor.padded_shape()};

    Program program{};

    IDevice* device{input_tensor.device()};

    auto src_buffer{input_tensor.buffer()};
    auto dst_buffer{output_tensor.buffer()};

    const auto dst_cb_data_format{datatype_to_dataformat_converter(output_tensor.dtype())};
    const bool fp32_dest_acc_en{
        (dst_cb_data_format == DataFormat::Float32) || (dst_cb_data_format == DataFormat::Int32) ||
        (dst_cb_data_format == DataFormat::UInt32)};
    const uint32_t height_tiles{input_shape[2] / constants::TILE_HEIGHT};
    const uint32_t width_tiles{input_shape[3] / constants::TILE_WIDTH};

    const uint32_t input_rank{input_tensor.padded_shape().rank()};

    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const int32_t dim{
        (operation_attributes.dim >= 0) ? operation_attributes.dim : (input_rank + operation_attributes.dim)};

    const uint32_t tiles_per_row{input_tensor.padded_shape()[dim]};
    const uint32_t num_rows_total{input_tensor.physical_volume() / tt::constants::TILE_HW / tiles_per_row};
    const uint32_t input_tile_offset{calc_input_tile_offset(input_shape, dim)};

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_rows_total);

    constexpr uint32_t in_tiles = 1;
    constexpr uint32_t one_tiles = 1;
    constexpr uint32_t intermed_tiles = 1;
    constexpr uint32_t out_tiles = 1;

    auto cb_src{create_cb(program, input_tensor.dtype(), CumprodCB::SRC, all_cores, in_tiles)};
    auto cb_acc{create_cb(program, input_tensor.dtype(), CumprodCB::ACC, all_cores, one_tiles)};
    auto cb_one{create_cb(program, input_tensor.dtype(), CumprodCB::ONE, all_cores, intermed_tiles)};
    auto cb_dst{create_cb(program, input_tensor.dtype(), CumprodCB::DST, all_cores, out_tiles)};

    const uint32_t src_is_dram{src_buffer->buffer_type() == BufferType::DRAM ? 1 : 0};
    const uint32_t dst_is_dram{dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0};

    const ReaderDataMovementConfig reader_config{{src_is_dram}};
    const ComputeConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false, .compile_args = {}};
    const WriterDataMovementConfig writer_config{{dst_is_dram}};

    auto cumprod_reader_kernel_id{create_kernel(program, KERNEL_PATHS[0], all_cores, reader_config)};
    auto cumprod_compute_sc_kernel_id{create_kernel(program, KERNEL_PATHS[1], core_group_1, compute_config)};
    std::optional<KernelHandle> compute_kernel_2_id{std::nullopt};
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{num_cols_per_core_group_2};
        compute_kernel_2_id = create_kernel(program, KERNEL_PATHS[1], core_group_2, compute_config);
    }
    auto cumprod_writer_kernel_id{create_kernel(program, KERNEL_PATHS[2], all_cores, writer_config)};

    for (uint32_t i{0}, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core{i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in any predefined core range.");
        }

        SetRuntimeArgs(
            program,
            cumprod_reader_kernel_id,
            core,
            {src_buffer->address(),
             num_tiles_per_core,
             tiles_per_row,
             input_tile_offset,
             tile_offset,
             tile_offset / input_tile_offset,
             tile_offset % input_tile_offset});

        SetRuntimeArgs(
            program,
            cumprod_writer_kernel_id,
            core,
            {dst_buffer->address(),
             num_tiles_per_core,
             tiles_per_row,
             input_tile_offset,
             tile_offset,
             tile_offset / input_tile_offset,
             tile_offset % input_tile_offset});

        if (core_group_1.contains(core)) {
            SetRuntimeArgs(program, cumprod_compute_sc_kernel_id, core, {num_tiles_per_core, tiles_per_row});
        } else if (core_group_2.contains(core)) {
            TT_ASSERT(compute_kernel_2_id.has_value());
            SetRuntimeArgs(program, compute_kernel_2_id.value(), core, {num_tiles_per_core, tiles_per_row});
        } else {
            TT_THROW("Core not in any predefined core range.");
        }

        tile_offset += num_tiles_per_core;
    }

    return {
        std::move(program),
        {.cumprod_reader_kernel_id = cumprod_reader_kernel_id,
         .cumprod_compute_kernel_id = cumprod_compute_sc_kernel_id,
         .cumprod_writer_kernel_id = cumprod_writer_kernel_id}};
}

void CumprodDeviceOperation::MultiCoreCumprodProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {}

CBHandle CumprodDeviceOperation::MultiCoreCumprodProgramFactory::create_cb(
    Program& program,
    const DataType& dtype,
    const CumprodCB& cumprod_cb,
    const CoreRangeSet& core_range_set,
    const uint32_t& num_tiles) {
    using tt::tt_metal::detail::TileSize;
    const uint32_t cb_id{static_cast<uint32_t>(cumprod_cb)};
    const auto cb_data_format{datatype_to_dataformat_converter(dtype)};
    const uint32_t single_tile_size{TileSize(cb_data_format)};
    const auto cb_config{CircularBufferConfig{num_tiles * single_tile_size, {{cb_id, cb_data_format}}}.set_page_size(
        cb_id, single_tile_size)};
    return CreateCircularBuffer(program, core_range_set, cb_config);
}

KernelHandle CumprodDeviceOperation::MultiCoreCumprodProgramFactory::create_kernel(
    Program& program,
    const char* kernel_path,
    const CoreRangeSet& core_range_set,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
    const std::vector<uint32_t>& runtime_args) {
    auto kernel_id{CreateKernel(program, kernel_path, core_range_set, config)};

    SetRuntimeArgs(program, kernel_id, core_range_set, runtime_args);

    return kernel_id;
}

}  // namespace ttnn::operations::experimental::reduction
