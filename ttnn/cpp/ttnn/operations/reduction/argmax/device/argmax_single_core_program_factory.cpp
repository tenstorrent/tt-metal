// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "ttnn/operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::reduction::detail {

using namespace tt::constants;

operation::ProgramWithCallbacks argmax_single_core(
    const Tensor& input, const Tensor& output, const std::optional<uint32_t> dim, const bool keepdim) {
    tt::tt_metal::Program program{};

    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_unit_size = input.element_size();
    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_unit_size = output.element_size();

    const tt::tt_metal::IDevice* device = output.device();

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_units = 1;  // single-core
    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_units);

    const auto& input_shape = input.padded_shape();
    const uint32_t rank = input_shape.size();
    const bool reduce_all = not dim.has_value();

    // Last dimension in input i.e. reduction dimension
    const uint32_t red_dim_units = input_shape[rank - 1];

    // Last dimension in output i.e. the dim left after reduction
    const auto output_last_dim = reduce_all or keepdim or (rank < 2) ? 1 : input_shape[rank - 2];

    // Create input CB to read reduction dim worth of data at once
    const uint32_t src_cb_idx = tt::CBIndex::c_0;
    const uint32_t src_page_size = round_up_to_mul32(red_dim_units * input_unit_size);
    tt::tt_metal::CircularBufferConfig src_cb_config =
        tt::tt_metal::CircularBufferConfig(src_page_size, {{src_cb_idx, input_cb_data_format}})
            .set_page_size(src_cb_idx, src_page_size);
    const auto src_cb = tt::tt_metal::CreateCircularBuffer(program, all_cores, src_cb_config);

    // Create output CB based on the output shape's last dimension
    const uint32_t dst_cb_idx = tt::CBIndex::c_1;
    const uint32_t dst_page_size = round_up_to_mul32(output_last_dim * output_unit_size);
    const tt::tt_metal::CircularBufferConfig dst_db_config =
        tt::tt_metal::CircularBufferConfig(dst_page_size, {{dst_cb_idx, output_cb_data_format}})
            .set_page_size(dst_cb_idx, dst_page_size);
    const auto dst_cb = tt::tt_metal::CreateCircularBuffer(program, all_cores, dst_db_config);

    const auto src_buffer = input.buffer();
    const auto dst_buffer = output.buffer();
    const bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const auto inner_dim_units = output_last_dim;
    const auto outer_dim_units = input.logical_volume() / inner_dim_units / red_dim_units;

    const std::vector<uint32_t> reader_compile_time_args = {
        src_cb_idx,
        dst_cb_idx,
        src_is_dram,
        dst_is_dram,
        src_page_size,
        dst_page_size,
        outer_dim_units,
        inner_dim_units,
        red_dim_units,
        (uint32_t)(reduce_all),
    };

    const std::map<std::string, std::string> kernel_defines;
    const tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, {src_buffer->address(), dst_buffer->address()});
    }

    auto override_runtime_args_callback = [reader_kernel_id, cores](
                                              const void* operation,
                                              const Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        for (const auto& core : cores) {
            {
                auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
                runtime_args[1] = dst_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::reduction::detail
