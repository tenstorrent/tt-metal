// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_op.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/util.hpp"

namespace ttnn::operations::data_movement {

operation::ProgramWithCallbacks indexed_fill_multi_core(const Tensor &batch_ids, const Tensor &input_a, const Tensor & input_b, const Tensor &output) {
    tt::tt_metal::Program program{};
    Device *device = input_a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto set_of_core_ranges = num_cores_to_corerange_set(num_cores_x*num_cores_y, compute_with_storage_grid_size);
    CoreRangeSet all_cores(set_of_core_ranges);


    uint32_t B = input_a.get_legacy_shape()[0];
    uint32_t b = input_b.get_legacy_shape()[0];

    TT_ASSERT(batch_ids.get_legacy_shape()[-1] == b);

    //parallelize across batch
    uint32_t num_units = B;
    uint32_t cb_index = 0;
    uint32_t batch_cb_index = 1;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_a.get_dtype());

    uint32_t page_size = input_a.get_legacy_shape()[-1] * input_a.element_size();
    uint32_t rounded_page_size = round_up_to_mul32(page_size);
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(2* rounded_page_size, {{cb_index, cb_data_format}})
            .set_page_size(cb_index, rounded_page_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    tt::DataFormat batch_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(batch_ids.get_dtype());
    uint32_t batch_page_size = round_up_to_mul32(b*sizeof(uint32_t));
    tt::tt_metal::CircularBufferConfig batch_cb_config =
        tt::tt_metal::CircularBufferConfig(2* batch_page_size, {{batch_cb_index, cb_data_format}})
            .set_page_size(batch_cb_index, batch_page_size);
    auto batch_cb = tt::tt_metal::CreateCircularBuffer(program, all_cores, batch_cb_config);


    bool in0_is_dram = input_a.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool in1_is_dram = input_b.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool batch_ids_is_dram = batch_ids.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(page_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)log2(page_size) : 0;

    // Create Kernels
    // reader
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)cb_index,
        (std::uint32_t)batch_cb_index,
        (std::uint32_t) batch_ids_is_dram,
        (std::uint32_t)in0_is_dram,
        (std::uint32_t)in1_is_dram,
        (std::uint32_t)stick_size_is_power_of_two,
        (std::uint32_t)log2_stick_size
    };

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/indexed_fill/device/kernels/dataflow/indexed_fill_reader.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(
            reader_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)cb_index,
        (std::uint32_t)out_is_dram,
        (std::uint32_t)stick_size_is_power_of_two,
        (std::uint32_t)log2_stick_size};

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto cores = grid_to_cores(num_cores_x*num_cores_y, num_cores_x, num_cores_y, false);

    uint32_t batch_size_in_sticks = input_a.get_legacy_shape()[1] * input_a.get_legacy_shape()[2];

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord &core = cores[i];
        uint32_t local_b = (i<B) ? b : 0;
        uint32_t local_batch_size_in_sticks = (i<B) ? batch_size_in_sticks : 0;

        std::vector<uint32_t> reader_runtime_args = {
                                                    batch_ids.buffer()->address(),
                                                    local_b,
                                                    input_a.buffer()->address(),
                                                    input_b.buffer()->address(),
                                                    page_size,
                                                    local_batch_size_in_sticks,
                                                    i
        };
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        std::vector<uint32_t> writer_runtime_args = {
                                                    output.buffer()->address(),
                                                    page_size,
                                                    local_batch_size_in_sticks,
                                                    i*local_batch_size_in_sticks
        };
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

    }
    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, cores,  page_size](
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        auto output = output_tensors.at(0);
        auto input_a = input_tensors.at(1);
        auto input_b = input_tensors.at(2);
        auto batch_ids = input_tensors.at(0);
        uint32_t core_id = 0;
        uint32_t B = input_a.get_legacy_shape()[0];
        uint32_t b = input_b.get_legacy_shape()[0];
        uint32_t batch_size_in_sticks = input_a.get_legacy_shape()[1] * input_a.get_legacy_shape()[2];
        for (const auto &core : cores) {
            uint32_t local_b = (core_id<B) ? b : 0;
            uint32_t local_batch_size_in_sticks = (core_id<B) ? batch_size_in_sticks : 0;
            std::vector<uint32_t> reader_runtime_args = {
                                                    batch_ids.buffer()->address(),
                                                    local_b,
                                                    input_a.buffer()->address(),
                                                    input_b.buffer()->address(),
                                                    page_size,
                                                    local_batch_size_in_sticks,
                                                    core_id
            };
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

            std::vector<uint32_t> writer_runtime_args = {
                                                    output.buffer()->address(),
                                                    page_size,
                                                    local_batch_size_in_sticks,
                                                    core_id*local_batch_size_in_sticks
            };

            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
            core_id++;

        }

    };
    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement
