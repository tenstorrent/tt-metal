// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include "bcast_to_device_operation.hpp"
#include "bcast_to_utils.hpp"

using namespace ttnn::operations::experimental::broadcast_to;

namespace ttnn::operations::experimental::broadcast_to {
using namespace tt::tt_metal;

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
    KernelHandle compute_kernel_id,
    CoreCoord compute_with_storage_grid_size,
    const BcastToOperation::operation_attributes_t& operation_attributes,
    const BcastToOperation::tensor_args_t& tensor_args,
    BcastToOperation::tensor_return_value_t& output,
    F handle_args) {
    const auto& input = tensor_args.input;

    const auto input_shape = input.padded_shape();

    const auto [iN, iC, iHt, iWt] = extract_shape_dims(input);
    const auto [oN, oC, oHt, oWt] = extract_shape_dims(output);

    uint32_t num_output_tiles = output.physical_volume() / output.tensor_spec().tile().get_tile_hw();

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
            handle_args(program, reader_kernel_id, core, std::array<uint32_t, 13>{0});
            handle_args(program, writer_kernel_id, core, std::array<uint32_t, 14>{0});
            handle_args(program, compute_kernel_id, core, std::array<uint32_t, 12>{0});
            continue;
        }

        uint32_t oHtWt = oHt * oWt;
        uint32_t tiles_per_batch = oHtWt * oC;
        uint32_t start_n = start_tile_id / tiles_per_batch;
        uint32_t start_remaining = start_tile_id % tiles_per_batch;
        uint32_t start_c = start_remaining / oHtWt;
        uint32_t start_t = start_remaining % oHtWt;
        uint32_t start_th = start_t / oWt;
        uint32_t start_tw = start_t % oWt;

        std::array reader_runtime_args = {
            input.buffer()->address(),
            start_n,
            start_c,
            start_t,
            start_th,
            start_tw,
            num_tiles_per_core,
            iHt * iWt * iC * (iN > 1),
            iHt * iWt * (iC > 1),
            oN,
            oC,
            oHt,
            oWt};
        handle_args(program, reader_kernel_id, core, reader_runtime_args);

        std::array writer_runtime_args = {
            output.buffer()->address(),
            start_n,
            start_c,
            start_t,
            start_th,
            start_tw,
            num_tiles_per_core,
            iHt * iWt * iC * (iN > 1),
            iHt * iWt * (iC > 1),
            oN,
            oC,
            oHt,
            oWt,
            start_tile_id};
        handle_args(program, writer_kernel_id, core, writer_runtime_args);

        std::array compute_runtime_args = {
            start_n,
            start_c,
            start_t,
            start_th,
            start_tw,
            num_tiles_per_core,
            iHt * iWt * iC * (iN > 1),
            iHt * iWt * (iC > 1),
            oN,
            oC,
            oHt,
            oWt};
        handle_args(program, compute_kernel_id, core, compute_runtime_args);

        start_tile_id += num_tiles_per_core;
    }
}

BcastToOperation::BcastToTileFactory::cached_program_t BcastToOperation::BcastToTileFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto input = tensor_args.input;
    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());

    auto output_shape = output.logical_shape();

    uint32_t input_single_tile_size = tt::tt_metal::detail::TileSize(input_data_format);

    // Device Setup
    auto* device = input.device();
    Program program = CreateProgram();

    // we parallelize the computation across the output tiles
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // How many tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    create_cb(tt::CBIndex::c_0, program, all_device_cores, input_single_tile_size, num_tiles_per_cb, input_data_format);

    create_cb(tt::CBIndex::c_1, program, all_device_cores, input_single_tile_size, num_tiles_per_cb, input_data_format);

    const auto src_is_dram = static_cast<const uint32_t>(input.buffer()->is_dram());
    const auto dst_is_dram = static_cast<const uint32_t>(output.buffer()->is_dram());

    auto kernel_config = BcastToKernelConfig(operation_attributes.subtile_broadcast_type);

    // READER KERNEL
    auto reader_id = tt::tt_metal::CreateKernel(
        program,
        get_kernel_file_path(kernel_config.reader_kernel),
        all_device_cores,
        tt::tt_metal::ReaderDataMovementConfig({src_is_dram, (uint32_t)tt::CBIndex::c_0}));

    // WRITER KERNEL
    auto writer_id = tt::tt_metal::CreateKernel(
        program,
        get_kernel_file_path(kernel_config.writer_kernel),
        all_device_cores,
        tt::tt_metal::WriterDataMovementConfig({dst_is_dram, (uint32_t)tt::CBIndex::c_1, (uint32_t)tt::CBIndex::c_0}));

    // COMPUTE KERNEL
    auto compute_id = tt::tt_metal::CreateKernel(
        program,
        get_kernel_file_path(kernel_config.compute_kernel),
        all_device_cores,
        tt::tt_metal::ComputeConfig{
            .math_approx_mode = false, .compile_args = {(uint32_t)tt::CBIndex::c_0, (uint32_t)tt::CBIndex::c_1}});

    auto set_runtime_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
        tt::tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
    };

    set_or_update_runtime_arguments(
        program,
        reader_id,
        writer_id,
        compute_id,
        compute_with_storage_grid_size,
        operation_attributes,
        tensor_args,
        output,
        set_runtime_args);

    return {std::move(program), {reader_id, writer_id, compute_id, compute_with_storage_grid_size}};
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
        cached_program.shared_variables.compute_kernel_id,
        cached_program.shared_variables.compute_with_storage_grid_size,
        operation_attributes,
        tensor_args,
        c,
        update_args);
}
}  // namespace ttnn::operations::experimental::broadcast_to
