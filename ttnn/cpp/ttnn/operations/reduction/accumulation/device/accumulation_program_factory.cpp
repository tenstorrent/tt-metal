// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "accumulation/device/accumulation_device_operation_types.hpp"
#include "accumulation_device_operation.hpp"

#include "tt-metalium/base_types.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <bit>

namespace ttnn::prim {

// calculate the offset between consecutive tiles between accumulation axis and last dimension
uint32_t AccumulationProgramFactory::calc_input_tile_offset(
    const Shape& input_shape, const int32_t& dim, uint32_t tile_height, uint32_t tile_width) {
    uint32_t input_tile_offset{1};
    for (int32_t i = dim + 1; i < input_shape.rank() - 2; ++i) {
        input_tile_offset *= input_shape[i];
    }
    if (input_shape.rank() > 1) {
        input_tile_offset *= (input_shape[-2] / tile_height);
    }
    if (input_shape.rank() > 0) {
        input_tile_offset *= (input_shape[-1] / tile_width);
    }

    return input_tile_offset;
}

AccumulationProgramFactory::cached_program_t AccumulationProgramFactory::create(
    const AccumulationParams& operation_attributes,
    const AccumulationInputs& tensor_args,
    Tensor& tensor_return_value) {
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

    const uint32_t input_rank{input_tensor.padded_shape().rank()};

    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;
    TT_FATAL(num_cores_y != 0, "Compute grid y-dimension must be non-zero");

    const int32_t dim{
        (operation_attributes.dim >= 0) ? operation_attributes.dim : (input_rank + operation_attributes.dim)};

    const auto& tile = input_tensor.tensor_spec().tile();
    // how many tiles along accumulation axis
    const uint32_t tiles_per_row{input_tensor.padded_shape()[dim]};
    TT_FATAL(tiles_per_row != 0, "tiles_per_row must be non-zero (got 0 for dim={})", dim);
    // all work units (product of all row lengths besides the accumulation row)
    const uint32_t num_rows_total{input_tensor.physical_volume() / tile.get_tile_hw() / tiles_per_row};
    // tiles between consecutive tiles along accumulation row
    const uint32_t input_tile_offset{calc_input_tile_offset(input_shape, dim, tile.get_height(), tile.get_width())};
    TT_FATAL(input_tile_offset != 0, "input_tile_offset must be non-zero (got 0 for dim={})", dim);

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_rows_total);

    constexpr uint32_t in_tiles = 4;
    constexpr uint32_t acc_tiles = 1;
    constexpr uint32_t out_tiles = 4;

    auto acc_dataformat = datatype_to_dataformat_converter(output_tensor.dtype());
    if (!is_integer_format(acc_dataformat)) {
        acc_dataformat = DataFormat::Float32;
    }

    const auto input_dataformat = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto output_dataformat = datatype_to_dataformat_converter(output_tensor.dtype());

    create_cb(program, input_dataformat, AccumulationCB::SRC, all_cores, in_tiles);
    create_cb(program, acc_dataformat, AccumulationCB::ACC, all_cores, acc_tiles);
    create_cb(program, output_dataformat, AccumulationCB::DST, all_cores, out_tiles);

    std::vector<UnpackToDestMode> unpack_to_dst(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dst[static_cast<unsigned>(AccumulationCB::ACC)] = UnpackToDestMode::UnpackToDestFp32;

    std::map<std::string, std::string> defines_kernel_args = {};

    if (is_integer_format(dst_cb_data_format)) {
        defines_kernel_args["BINARY_OP_INIT"] =
            operation_attributes.op == AccumulationOp::CUMSUM ? "add_int_tile_init" : "mul_int_tile_init";
        defines_kernel_args["BINARY_OP"] = operation_attributes.op == AccumulationOp::CUMSUM
                                               ? "add_int_tile<DataFormat::Int32>"
                                               : "mul_int_tile<DataFormat::Int32>";
        unpack_to_dst[static_cast<unsigned>(AccumulationCB::SRC)] = UnpackToDestMode::UnpackToDestFp32;
    } else {
        defines_kernel_args["BINARY_OP_INIT"] =
            operation_attributes.op == AccumulationOp::CUMSUM ? "add_binary_tile_init" : "mul_binary_tile_init";
        defines_kernel_args["BINARY_OP"] =
            operation_attributes.op == AccumulationOp::CUMSUM ? "add_binary_tile" : "mul_binary_tile";
        unpack_to_dst[static_cast<unsigned>(AccumulationCB::SRC)] = UnpackToDestMode::UnpackToDestFp32;
    }

    float default_acc_value = 0.f;
    if (operation_attributes.op == AccumulationOp::CUMPROD) {
        default_acc_value = 1.f;
        if (is_integer_format(dst_cb_data_format)) {
            default_acc_value = std::bit_cast<float>(1U);
        }
    }

    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(reader_compile_time_args);
    const ReaderDataMovementConfig reader_config{reader_compile_time_args};
    // Due to hardware bug (#38306), HiFi4 + fp32_dest_acc_en can sometime produce incorrect results on Wormhole.
    // Use HiFi3 silently when fp32_dest_acc_en is True on Wormhole B0.
    const ComputeConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = unpack_to_dst,
        .math_approx_mode = false,
        .compile_args = {std::bit_cast<uint32_t>(default_acc_value)},
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
             static_cast<uint32_t>(operation_attributes.flip)});

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
            SetRuntimeArgs(program, accumulation_compute_kernel_id, core, {num_tiles_per_core, tiles_per_row});
        } else if (core_group_2.contains(core)) {
            TT_ASSERT(compute_kernel_2_id.has_value());
            SetRuntimeArgs(program, compute_kernel_2_id.value(), core, {num_tiles_per_core, tiles_per_row});
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
    const AccumulationParams& /*operation_attributes*/,
    const AccumulationInputs& tensor_args,
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
    const tt::DataFormat& data_format,
    const AccumulationCB& accumulation_cb,
    const CoreRangeSet& core_range_set,
    const uint32_t& num_tiles) {
    const uint32_t cb_id{static_cast<uint32_t>(accumulation_cb)};

    const uint32_t single_tile_size{tt::tile_size(data_format)};
    const auto cb_config{CircularBufferConfig{num_tiles * single_tile_size, {{cb_id, data_format}}}.set_page_size(
        cb_id, single_tile_size)};
    return CreateCircularBuffer(program, core_range_set, cb_config);
}

KernelHandle AccumulationProgramFactory::create_kernel(
    Program& program,
    const char* kernel_path,
    const CoreRangeSet& core_range_set,
    const std::variant<DataMovementConfig, ComputeConfig>& config,
    const std::vector<uint32_t>& runtime_args) {
    auto kernel_id{CreateKernel(program, kernel_path, core_range_set, config)};

    SetRuntimeArgs(program, kernel_id, core_range_set, runtime_args);

    return kernel_id;
}

}  // namespace ttnn::prim
