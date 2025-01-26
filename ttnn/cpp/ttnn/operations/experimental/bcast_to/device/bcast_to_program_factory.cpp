// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <vector>

#include "common/tt_backend_api_types.hpp"
#include "common/work_split.hpp"
#include "bcast_to_device_operation.hpp"
#include "host_api.hpp"
#include "hostdevcommon/kernel_structs.h"
#include "impl/buffers/buffer.hpp"
#include "impl/buffers/circular_buffer_types.hpp"
#include "impl/kernels/kernel_types.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "bcast_to_utils.hpp"

using namespace tt::tt_metal;
using namespace ttnn::operations::experimental::broadcast_to;

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_shape_dims(const Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
}

template <typename F>
void set_or_update_runtime_arguments(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle writer_kernel_id,
    CoreCoord compute_with_storage_grid_size,
    const BcastToOperation::operation_attributes_t& operation_attributes,
    const BcastToOperation::tensor_args_t& tensor_args,
    BcastToOperation::tensor_return_value_t& output,
    F handle_args) {
    const auto& input = tensor_args.input;

    const auto input_shape = input.padded_shape();
    const auto output_shape = output.padded_shape();

    const auto [iN, iC, iHt, iWt] = extract_shape_dims(input);
    const auto [oN, oC, oHt, oWt] = extract_shape_dims(output);

    uint32_t num_output_tiles = output.volume() / output.tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            handle_args(program, reader_kernel_id, core, std::array<uint32_t, 10>{0});
            handle_args(program, writer_kernel_id, core, std::array<uint32_t, 11>{0});
            continue;
        }

        uint32_t oHtWt = oHt * oWt;
        std::array reader_runtime_args = {
            input.buffer()->address(),
            start_tile_id,
            num_tiles_per_core,
            oHtWt,
            iHt * iWt * iC * (iN > 1),
            iHt * iWt * (iC > 1),
            oN,
            oC,
            oHt,
            oWt};
        handle_args(program, reader_kernel_id, core, reader_runtime_args);

        std::array writer_runtime_args = {
            output.buffer()->address(), start_tile_id, num_tiles_per_core, oHtWt, oN, oC, oHt, oWt, 0u, 0u};
        handle_args(program, writer_kernel_id, core, writer_runtime_args);

        start_tile_id += num_tiles_per_core;
    }
}

namespace ttnn::operations::experimental::broadcast_to {
BcastToOperation::BcastToTileFactory::cached_program_t BcastToOperation::BcastToTileFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto input = tensor_args.input;
    auto input_shape = input.get_shape();
    uint32_t data_size = input.element_size();
    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.get_dtype());

    auto output_shape = output.get_shape();
    auto output_data_format = datatype_to_dataformat_converter(output.get_dtype());

    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_data_format);
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_data_format);
    uint32_t num_output_tiles = output.volume() / output.tensor_spec().tile().get_tile_hw();

    // Device Setup
    auto* device = input.device();
    Program program = CreateProgram();

    // we parallelize the computation across the output tiles
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

#ifdef DEBUG
    tt::log_debug(tt::LogOp, "Data size = {}\n", data_size);

    // tt::log_debug("Input Page size = %lu\n", input.buffer()->page_size());
    // tt::log_debug("Output Page size = %lu\n", output.buffer()->page_size());

    std::stringstream debug_stream;

    debug_stream << "Input Shape = ";
    for (auto i = 0; i < input_shape.size(); i++) {
        debug_stream << input_shape[i] << " ";
    }
    debug_stream << std::endl;

    debug_stream << "Output Shape = ";
    for (auto i = 0; i < output_shape.size(); i++) {
        debug_stream << output_shape[i] << " ";
    }
    debug_stream << std::endl;

    tt::log_debug(tt::LogOp, "{}", debug_stream.str().c_str());

#endif

    // How many tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    auto [input_cb, input_cb_handle] = create_cb(
        tt::CBIndex::c_0, program, all_device_cores, input_single_tile_size, num_tiles_per_cb, input_data_format);

    const auto src_is_dram = static_cast<const uint32_t>(input.buffer()->is_dram());
    const auto dst_is_dram = static_cast<const uint32_t>(output.buffer()->is_dram());

    auto kernel_config = BcastToKernelConfig(operation_attributes.subtile_broadcast_type);

    // READER KERNEL
    auto reader_id = tt::tt_metal::CreateKernel(
        program,
        get_kernel_file_path(kernel_config.reader_kernel),
        all_device_cores,
        tt::tt_metal::ReaderDataMovementConfig({src_is_dram}));

    // WRITER KERNEL
    auto writer_id = tt::tt_metal::CreateKernel(
        program,
        get_kernel_file_path(kernel_config.writer_kernel),
        all_device_cores,
        tt::tt_metal::WriterDataMovementConfig({dst_is_dram}));

    auto set_runtime_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
        tt::tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
    };

    set_or_update_runtime_arguments(
        program,
        reader_id,
        writer_id,
        compute_with_storage_grid_size,
        operation_attributes,
        tensor_args,
        output,
        set_runtime_args);

    return {std::move(program), {reader_id, writer_id, compute_with_storage_grid_size}};
}

void BcastToOperation::BcastToTileFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& c) {
    auto update_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
        auto& all_args = GetRuntimeArgs(program, kernel_id);
        auto& core_args = all_args.at(core.x).at(core.y);
        std::copy(args.begin(), args.end(), core_args.data());
    };

    set_or_update_runtime_arguments(
        cached_program.program,
        cached_program.shared_variables.reader_kernel_id,
        cached_program.shared_variables.writer_kernel_id,
        cached_program.shared_variables.compute_with_storage_grid_size,
        operation_attributes,
        tensor_args,
        c,
        update_args);
}
}  // namespace ttnn::operations::experimental::broadcast_to
