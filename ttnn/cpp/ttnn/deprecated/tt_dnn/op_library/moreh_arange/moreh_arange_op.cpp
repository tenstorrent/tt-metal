// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include "ttnn/deprecated/tt_dnn/op_library/moreh_arange/moreh_arange_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/test_tiles.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

operation::ProgramWithCallbacks moreh_arange_(
    const Tensor &output, float start, float end, float step, bool untilize_out, const CoreRange core_range) {
    // split work
    // N and C are always 1
    // H is always TILE_HEIGHT
    auto shape = output.get_legacy_shape();
    auto W = shape[-1];
    auto Wt = div_up(W, TILE_WIDTH);

    uint32_t units_to_divide = Wt;
    uint32_t core_w = core_range.end_coord.x - core_range.start_coord.x + 1;
    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(core_range, units_to_divide);

    auto element_size = output.element_size();

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_out0, 1},  // output
        });

    // create read/wrtie kernel
    bool dst_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::map<string, string> writer_defines;

    if (output.get_dtype() == DataType::BFLOAT16) {
        writer_defines["OUTPUT_DTYPE_BFLOAT16"] = 1;
    }
    if (output.get_dtype() == DataType::INT32) {
        writer_defines["OUTPUT_DTYPE_INT32"] = 1;
    }
    if (output.get_dtype() == DataType::FLOAT32) {
        writer_defines["OUTPUT_DTYPE_FLOAT32"] = 1;
    }

    auto kernel_id = CreateWriteKernel(
        program,
        untilize_out ? "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_arange/kernels/writer_moreh_arange_rm.cpp"
                     : "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_arange/kernels/writer_moreh_arange.cpp",
        all_cores,
        {dst_is_dram},
        writer_defines);

    // Set Runtime Args
    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        vector<uint32_t> writer_args = {
            output.buffer()->address(),
            tile_offset,
            num_tiles_per_core,
            *reinterpret_cast<uint32_t *>(&start),
            *reinterpret_cast<uint32_t *>(&step),
            element_size};

        SetRuntimeArgs(program, kernel_id, core, writer_args);

        tile_offset += num_tiles_per_core;
    }

    auto override_runtime_args_callback = [kernel_id = kernel_id, num_cores, core_h](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        TT_FATAL(output_buffers.size() == 1);

        auto src_dram_buffer = output_buffers.at(0);

        for (uint32_t icore = 0; icore < num_cores; icore++) {
            CoreCoord core = {icore / core_h, icore % core_h};

            {
                auto &runtime_args = GetRuntimeArgs(program, kernel_id, core);
                runtime_args[0] = src_dram_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void MorehArange::validate_with_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    TT_FATAL(this->step > 0 || this->step < 0, "step must be nonzero");
    TT_FATAL(
        ((this->step > 0) && (this->end >= this->start)) || ((this->step < 0) && (this->end <= this->start)),
        "upper bound and larger bound inconsistent with step sign");

    auto output_dtype_has_value = this->output_dtype.has_value();

    if (output_dtype_has_value) {
        auto output_dtype = this->output_dtype.value();
        TT_FATAL(output_dtype != DataType::BFLOAT8_B, "moreh arange not support bfloat8_b dtype");
        TT_FATAL(output_dtype != DataType::UINT32, "moreh arange not support uint32 dtype");
    }

    if (output_tensors.empty() || !output_tensors.at(0).has_value()) {
        // If the user decided to not use any optional output tensors, then this would be empty or would be a nullptr.
        return;
    }
    TT_FATAL(output_tensors.size() == 1, "Must have 1 output tensor");

    auto output_tensor = output_tensors.front().value();
    auto output_memory_layout = output_tensor.memory_config().memory_layout;
    auto output_layout = output_tensor.get_layout();

    if (output_dtype_has_value) {
        auto output_dtype = this->output_dtype.value();
        TT_FATAL(output_dtype == output_tensor.get_dtype(), "If output_tensor is provided as input, its dtype should match the output_dtype parameter.");
    }

    TT_FATAL(output_memory_layout == TensorMemoryLayout::INTERLEAVED);

    if (this->untilize_out) {
        TT_FATAL(output_layout == Layout::ROW_MAJOR);
    } else {
        TT_FATAL(output_layout == Layout::TILE);
    }

}

std::vector<Shape> MorehArange::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    // return size is ceil((end - start) / step)
    uint32_t num_elems = static_cast<uint32_t>(ceil(((this->end - this->start) / this->step)));

    if (this->untilize_out) {
        Shape output_shape = {num_elems};

        return {output_shape};
    }

    std::vector<uint32_t> output_size_vec = {TILE_HEIGHT, round_up(num_elems, TILE_WIDTH)};

    auto dimensions_pads = std::vector<Padding::PadDimension>();
    dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = 31});
    dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = round_up(num_elems, TILE_WIDTH) - num_elems});

    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    auto output_shape = Shape(output_size_vec, padding);

    return {output_shape};
}

std::vector<Tensor> MorehArange::create_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    if (!output_tensors.empty() && output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    // default dtype is bfloat16
    auto output_dtype = DataType::BFLOAT16;
    if (this->output_dtype.has_value()) {
        output_dtype = this->output_dtype.value();
    }

    auto layout = (this->untilize_out) ? Layout::ROW_MAJOR : Layout::TILE;
    return operation::generic_create_output_tensors(
        *this, input_tensors, output_dtype, layout, this->output_mem_config);
}

operation::ProgramWithCallbacks MorehArange::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    return moreh_arange_(
        output_tensors.at(0), this->start, this->end, this->step, this->untilize_out, this->core_range);
}

Tensor moreh_arange(
    float start,
    float end,
    float step,
    const Tensor &any,
    std::optional<Tensor> output_tensor,
    bool untilize_out,
    std::optional<DataType> output_dtype,
    const MemoryConfig &output_mem_config) {
    auto device = any.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({any}))};

    operation::launch_op(
        [start, end, step, untilize_out, output_dtype, all_cores, output_mem_config](
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehArange{
                    .start = start,
                    .end = end,
                    .step = step,
                    .untilize_out = untilize_out,
                    .output_dtype = output_dtype,
                    .core_range = all_cores,
                    .output_mem_config = output_mem_config,
                },
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {any},
        output_tensors,
        {},
        {output_tensor});

    return output_tensors.at(0);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
