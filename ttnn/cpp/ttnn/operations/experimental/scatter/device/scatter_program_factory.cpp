// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_program_factory.hpp"

#include "scatter_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

namespace ttnn::operations::experimental::scatter {

ScatterProgramFactory::cached_program_t ScatterProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;
    using namespace tt::constants;

    Program program{};

    const auto& input_tensor{tensor_args.input_tensor};
    const auto& input_shape{input_tensor.get_padded_shape()};
    const auto& input_rank{input_shape.rank()};
    const auto& index_tensor{tensor_args.index_tensor};
    const auto& index_shape{index_tensor.get_padded_shape()};
    const auto& index_rank{index_shape.rank()};
    const auto& src_tensor{tensor_args.src_tensor};
    const auto& src_shape{src_tensor.get_padded_shape()};
    const auto& src_rank{src_shape.rank()};

    const tt::DataFormat input_tensor_cb_data_format = datatype_to_dataformat_converter(input_tensor.get_dtype());
    const tt::DataFormat index_tensor_cb_data_format = datatype_to_dataformat_converter(index_tensor.get_dtype());
    const tt::DataFormat src_tensor_cb_data_format = datatype_to_dataformat_converter(src_tensor.get_dtype());
    const tt::DataFormat output_tensor_cb_data_format = datatype_to_dataformat_converter(output_tensor.get_dtype());

    const uint32_t input_tensor_tile_size = tile_size(input_tensor_cb_data_format);
    const uint32_t index_tensor_tile_size = tile_size(index_tensor_cb_data_format);
    const uint32_t src_tensor_tile_size = tile_size(src_tensor_cb_data_format);
    const uint32_t output_tensor_tile_size = tile_size(output_tensor_cb_data_format);

    auto input_buffer = input_tensor.buffer();
    auto index_buffer = index_tensor.buffer();
    auto src_buffer = src_tensor.buffer();
    auto output_buffer = output_tensor.buffer();

    const uint32_t input_tensor_is_dram = input_buffer->buffer_type() == BufferType::DRAM;
    const uint32_t index_tensor_is_dram = index_buffer->buffer_type() == BufferType::DRAM;
    const uint32_t src_tensor_is_dram = src_buffer->buffer_type() == BufferType::DRAM;
    const uint32_t output_tensor_is_dram = output_buffer->buffer_type() == BufferType::DRAM;

    const uint32_t num_input_tiles = input_tensor.volume() / TILE_HW;
    const uint32_t num_index_tiles = index_tensor.volume() / TILE_HW;
    const uint32_t num_src_tiles = src_tensor.volume() / TILE_HW;
    const uint32_t num_output_tiles = output_tensor.volume() / TILE_HW;

    const uint32_t logical_index_height = index_shape[0] * index_shape[1] * index_shape[2];
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / TILE_HEIGHT;
    const uint32_t Wt_input = input_shape[3] / TILE_WIDTH;
    const uint32_t Wt_index = index_shape[3] / TILE_WIDTH;

    constexpr uint32_t num_cb_unit = 2;

    const int32_t dim{(args.dim >= 0) ? args.dim : (input_rank + args.dim)};

    auto device = input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t total_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    const uint32_t all_core_utilization_loop_count = Ht / total_number_of_cores;
    const uint32_t all_core_utilization_loop_remainder = Ht % total_number_of_cores;

    const CoreCoord core{0, 0};

    auto grid{device->compute_with_storage_grid_size()};
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, Ht);

    const uint32_t input_tiles = num_cb_unit * Wt_input;
    const uint32_t index_tiles = num_cb_unit * Wt_index;
    const uint32_t src_tiles = num_cb_unit * Wt_index;
    const uint32_t out_tiles = num_cb_unit * Wt_input;

    auto cb_input{create_cb(program, input_tensor.get_dtype(), ScatterCB::INPUT, all_cores, input_tiles)};
    auto cb_index{create_cb(program, index_tensor.get_dtype(), ScatterCB::INDEX, all_cores, index_tiles)};
    auto cb_src{create_cb(program, src_tensor.get_dtype(), ScatterCB::SRC, all_cores, src_tiles)};
    auto cb_dst{create_cb(program, output_tensor.get_dtype(), ScatterCB::DST, all_cores, out_tiles)};

    constexpr const char* reader_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/scatter/device/kernels/dataflow/reader_scatter.cpp";
    constexpr const char* writer_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/scatter/device/kernels/dataflow/writer_scatter.cpp";

    const std::vector<uint32_t> compile_time_args{
        {input_tensor_is_dram,
         index_tensor_is_dram,
         src_tensor_is_dram,
         output_tensor_is_dram,
         input_tensor.buffer()->address(),
         index_tensor.buffer()->address(),
         src_tensor.buffer()->address(),
         output_tensor.buffer()->address(),
         static_cast<uint32_t>(ScatterCB::INPUT),
         static_cast<uint32_t>(ScatterCB::INDEX),
         static_cast<uint32_t>(ScatterCB::SRC),
         static_cast<uint32_t>(ScatterCB::DST),
         Wt_input,
         index_tensor.get_logical_shape()[-1],
         logical_index_height,
         Wt_index,
         Ht,
         total_number_of_cores,
         compute_with_storage_grid_size.x}};

    auto reader_kernel =
        create_kernel(program, reader_kernel_path, all_cores, ReaderDataMovementConfig{compile_time_args});
    auto writer_kernel =
        create_kernel(program, writer_kernel_path, all_cores, WriterDataMovementConfig{compile_time_args});

    const uint32_t& num_cores_y = compute_with_storage_grid_size.y;
    uint32_t tile_offset = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core{i / num_cores_y, i % num_cores_y};

        uint32_t ht_per_core;
        if (core_group_1.contains(core)) {
            ht_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            ht_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in any predefined core range.");
        }

        SetRuntimeArgs(program, reader_kernel, core, {tile_offset, ht_per_core});

        SetRuntimeArgs(program, writer_kernel, core, {tile_offset, ht_per_core});

        // if (core_group_1.contains(core)) {
        //     SetRuntimeArgs(program, cumprod_compute_sc_kernel_id, core, {num_tiles_per_core, tiles_per_row});
        // } else if (core_group_2.contains(core)) {
        //     TT_ASSERT(compute_kernel_2_id.has_value());
        //     SetRuntimeArgs(program, compute_kernel_2_id.value(), core, {num_tiles_per_core, tiles_per_row});
        // } else {
        //     TT_THROW("Core not in any predefined core range.");
        // }

        tile_offset += ht_per_core;
    }

    return {std::move(program), {reader_kernel, writer_kernel, compute_with_storage_grid_size}};
}

void ScatterProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {}

CBHandle ScatterProgramFactory::create_cb(
    Program& program,
    const DataType& dtype,
    const ScatterCB& scatter_cb,
    const CoreRangeSet& core_range_set,
    const uint32_t& num_tiles) {
    using tt::tt_metal::detail::TileSize;
    const uint32_t cb_id{static_cast<uint32_t>(scatter_cb)};
    const auto cb_data_format{datatype_to_dataformat_converter(dtype)};
    const uint32_t single_tile_size{TileSize(cb_data_format)};
    const auto cb_config{CircularBufferConfig{num_tiles * single_tile_size, {{cb_id, cb_data_format}}}.set_page_size(
        cb_id, single_tile_size)};
    return CreateCircularBuffer(program, core_range_set, cb_config);
}

KernelHandle ScatterProgramFactory::create_kernel(
    Program& program,
    const char* kernel_path,
    const CoreRangeSet& core_range_set,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
    const std::vector<uint32_t>& runtime_args) {
    auto kernel_id{CreateKernel(program, kernel_path, core_range_set, config)};

    if (!runtime_args.empty()) {
        SetRuntimeArgs(program, kernel_id, core_range_set, runtime_args);
    }

    return kernel_id;
}

}  // namespace ttnn::operations::experimental::scatter
