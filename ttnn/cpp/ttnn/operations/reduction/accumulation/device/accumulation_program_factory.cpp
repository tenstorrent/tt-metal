// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "accumulation_device_operation.hpp"

#include "tt-metalium/base_types.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::reduction::accumulation {

// calculate the offset between consecutive tiles between accumulation axis and last dimension
uint32_t AccumulationProgramFactory::calc_input_tile_offset(const Shape& input_shape, const int32_t& dim) {
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

AccumulationProgramFactory::cached_program_t AccumulationProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor{tensor_args.input_tensor};
    auto& output_tensor{tensor_return_value};
    const auto& input_shape{input_tensor.padded_shape()};

    Program program{};

    IDevice* device{input_tensor.device()};

    auto* src_buffer{input_tensor.buffer()};
    auto* dst_buffer{output_tensor.buffer()};

    const auto dst_cb_data_format{datatype_to_dataformat_converter(output_tensor.dtype())};
    const bool fp32_dest_acc_en{
        (dst_cb_data_format == DataFormat::Float32) || (dst_cb_data_format == DataFormat::Int32) ||
        (dst_cb_data_format == DataFormat::UInt32)};

    const uint32_t input_rank{input_tensor.padded_shape().rank()};

    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const int32_t dim{
        (operation_attributes.dim >= 0) ? operation_attributes.dim : (input_rank + operation_attributes.dim)};

    const auto& tile = input_tensor.tensor_spec().tile();
    // how many tiles along accumulation axis
    const uint32_t tiles_per_row{input_tensor.padded_shape()[dim]};
    // all work units (product of all row lengths besides the accumulation row)
    const uint32_t num_rows_total{input_tensor.physical_volume() / tile.get_tile_hw() / tiles_per_row};
    // tiles between consecutive tiles along accumulation row
    const uint32_t input_tile_offset{calc_input_tile_offset(input_shape, dim)};

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_rows_total);

    constexpr uint32_t in_tiles = 4;
    constexpr uint32_t op_tiles = 4;
    constexpr uint32_t start_tiles = 4;
    constexpr uint32_t out_tiles = 4;

    create_cb(program, input_tensor.dtype(), AccumulationCB::SRC, all_cores, in_tiles);
    create_cb(program, output_tensor.dtype(), AccumulationCB::ACC, all_cores, op_tiles);
    create_cb(program, output_tensor.dtype(), AccumulationCB::START, all_cores, start_tiles);
    create_cb(program, output_tensor.dtype(), AccumulationCB::DST, all_cores, out_tiles);

    std::map<std::string, std::string> defines_kernel_args = {};
    if (is_integer_format(dst_cb_data_format)) {
        // Used to switch to add_tile_int32() instead of add_tiles()
        defines_kernel_args["CUMSUM_USE_INT32"] = "1";
    }

    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(reader_compile_time_args);
    const ReaderDataMovementConfig reader_config{reader_compile_time_args};
    const ComputeConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = false,
        .compile_args = {},
        .defines = defines_kernel_args};

    std::vector<uint32_t> writer_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args);
    const WriterDataMovementConfig writer_config{writer_compile_time_args};

    auto accumulation_reader_kernel_id{create_kernel(program, KERNEL_PATHS[0], all_cores, reader_config)};
    auto accumulation_compute_kernel_id{create_kernel(program, KERNEL_PATHS[1], core_group_1, compute_config)};
    std::optional<KernelHandle> compute_kernel_2_id{std::nullopt};
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{num_cols_per_core_group_2};
        compute_kernel_2_id = create_kernel(program, KERNEL_PATHS[1], core_group_2, compute_config);
    }
    auto accumulation_writer_kernel_id{create_kernel(program, KERNEL_PATHS[2], all_cores, writer_config)};

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
            accumulation_reader_kernel_id,
            core,
            {src_buffer->address(),
             num_tiles_per_core,
             tiles_per_row,
             input_tile_offset,
             tile_offset,
             tile_offset / input_tile_offset,
             tile_offset % input_tile_offset,
             static_cast<uint32_t>(operation_attributes.flip),
             static_cast<uint32_t>(operation_attributes.op)});

        SetRuntimeArgs(
            program,
            accumulation_writer_kernel_id,
            core,
            {dst_buffer->address(),
             num_tiles_per_core,
             tiles_per_row,
             input_tile_offset,
             tile_offset,
             tile_offset / input_tile_offset,
             tile_offset % input_tile_offset,
             static_cast<uint32_t>(operation_attributes.flip)});

        if (core_group_1.contains(core)) {
            SetRuntimeArgs(
                program,
                accumulation_compute_kernel_id,
                core,
                {num_tiles_per_core, tiles_per_row, static_cast<uint32_t>(operation_attributes.op)});
        } else if (core_group_2.contains(core)) {
            TT_ASSERT(compute_kernel_2_id.has_value());
            SetRuntimeArgs(
                program,
                compute_kernel_2_id.value(),
                core,
                {num_tiles_per_core, tiles_per_row, static_cast<uint32_t>(operation_attributes.op)});
        } else {
            TT_THROW("Core not in any predefined core range.");
        }

        tile_offset += num_tiles_per_core;
    }

    auto cores = grid_to_cores(num_cores, grid.x, grid.y);
    return {
        std::move(program),
        {.accumulation_reader_kernel_id = accumulation_reader_kernel_id,
         .accumulation_compute_kernel_id = accumulation_compute_kernel_id,
         .accumulation_compute_kernel_id_2 = compute_kernel_2_id,
         .accumulation_writer_kernel_id = accumulation_writer_kernel_id,
         .cores = cores}};
}

void AccumulationProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    const auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.accumulation_reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.accumulation_writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    auto input_buffer_address = tensor_args.input_tensor.buffer()->address();
    auto output_buffer_address = tensor_return_value.buffer()->address();
    for (const auto& core : cores) {
        auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
        reader_runtime_args[0] = input_buffer_address;
        writer_runtime_args[0] = output_buffer_address;
    }
}

CBHandle AccumulationProgramFactory::create_cb(
    Program& program,
    const DataType& dtype,
    const AccumulationCB& accumulation_cb,
    const CoreRangeSet& core_range_set,
    const uint32_t& num_tiles) {
    const uint32_t cb_id{static_cast<uint32_t>(accumulation_cb)};
    const auto cb_data_format{datatype_to_dataformat_converter(dtype)};
    const uint32_t single_tile_size{tt::tile_size(cb_data_format)};
    const auto cb_config{CircularBufferConfig{num_tiles * single_tile_size, {{cb_id, cb_data_format}}}.set_page_size(
        cb_id, single_tile_size)};
    return CreateCircularBuffer(program, core_range_set, cb_config);
}

KernelHandle AccumulationProgramFactory::create_kernel(
    Program& program,
    const char* kernel_path,
    const CoreRangeSet& core_range_set,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
    const std::vector<uint32_t>& runtime_args) {
    auto kernel_id{CreateKernel(program, kernel_path, core_range_set, config)};

    SetRuntimeArgs(program, kernel_id, core_range_set, runtime_args);

    return kernel_id;
}

}  // namespace ttnn::operations::reduction::accumulation
