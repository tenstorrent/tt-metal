// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/device/concat_program_factory.hpp"

#include <algorithm>
#include <numeric>

#include "ttnn/operations/data_movement/concat/device/concat_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/tt_align.hpp>

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

uint32_t find_greatest_common_page_size(std::vector<uint32_t>& stick_sizes, uint32_t alignment) {
    TT_FATAL(stick_sizes.size() > 0, "Need at least one stick size to find page size");
    uint32_t page_size = tt::align(stick_sizes[0], alignment);
    for (size_t idx = 1; idx < stick_sizes.size(); idx++) {
        const uint32_t padded_stick_size = tt::align(stick_sizes[idx], alignment);
        page_size = std::gcd(page_size, padded_stick_size);
    }
    return page_size;
}

}  // namespace

namespace ttnn::operations::data_movement::detail {

tt_metal::operation::ProgramWithCallbacks s2s_tiled_concat_two_tensors_height_multi_core(
    const std::vector<Tensor>& input_tensors, uint32_t dim, Tensor& output, unsigned int groups) {
    // If we end up here for concat with more than 2 tensors on any other dim we should have
    // taken another path
    TT_FATAL(dim == 3, "Sharded concat with tiled inputs only supports dim=3 (was {})", dim);
    TT_FATAL(input_tensors.size() == 2, "Expected 2 input tensors (was {})", input_tensors.size());

    TT_FATAL(
        input_tensors[0].logical_shape()[-1] == input_tensors[0].padded_shape()[-1],
        "Cannot have padding along width dimension in input tensor 0 ({} != {})",
        input_tensors[0].logical_shape()[-1],
        input_tensors[0].padded_shape()[-1]);
    TT_FATAL(
        input_tensors[1].logical_shape()[-1] == input_tensors[1].padded_shape()[-1],
        "Cannot have padding along width dimension in input tensor 1 ({} != {})",
        input_tensors[1].logical_shape()[-1],
        input_tensors[1].padded_shape()[-1]);

    TT_FATAL(
        input_tensors[0].padded_shape()[-1] % groups == 0,
        "Input tensor 0 columns must be evenly divisible by groups (W={}, groups={})",
        input_tensors[0].padded_shape()[-1],
        groups);
    TT_FATAL(
        input_tensors[1].padded_shape()[-1] % groups == 0,
        "Input tensor 1 columns must be evenly divisible by groups (W={}, groups={})",
        input_tensors[1].padded_shape()[-1],
        groups);

    // The current implementation relies on not having break up tile faces so if we would
    // need to split tiles because dim[-1] / groups < 16, we cannot proceed
    TT_FATAL(
        input_tensors[0].padded_shape()[-1] / groups >= TILE_HEIGHT / 2,
        "Group size must be at least 16 for input0 (was {})",
        input_tensors[0].padded_shape()[-1] / groups);
    TT_FATAL(
        input_tensors[1].padded_shape()[-1] / groups >= TILE_HEIGHT / 2,
        "Group size must be at least 16 for input1 (was {})",
        input_tensors[1].padded_shape()[-1] / groups);

    tt_metal::Program program = tt_metal::CreateProgram();

    const auto all_cores = input_tensors[0].shard_spec().value().grid;  // assume all inputs have same grid

    const auto get_num_tiles_per_shard =
        [](const std::array<uint32_t, 2>& shard_shape) -> std::tuple<uint32_t, uint32_t> {
        TT_FATAL(shard_shape[0] % TILE_HEIGHT == 0, "Shard height must be aligned to tile height");
        TT_FATAL(shard_shape[1] % TILE_WIDTH == 0, "Shard width must be aligned to tile width");
        const uint32_t num_tiles_along_height = shard_shape[0] / TILE_HEIGHT;
        const uint32_t num_tiles_along_width = shard_shape[1] / TILE_WIDTH;
        TT_FATAL(num_tiles_along_height != 0 && num_tiles_along_width != 0, "Expected tensor to have at least 1 tiles");
        return {num_tiles_along_height, num_tiles_along_width};
    };

    const auto get_total_num_tiles_per_shard = [](const std::tuple<uint32_t, uint32_t>& num_tiles) -> uint32_t {
        return std::get<0>(num_tiles) * std::get<1>(num_tiles);
    };

    std::vector<std::tuple<uint32_t, uint32_t>> num_tiles_for_each_input_shard;
    num_tiles_for_each_input_shard.reserve(input_tensors.size());
    for (const auto& input_tensor : input_tensors) {
        num_tiles_for_each_input_shard.push_back(get_num_tiles_per_shard(input_tensor.shard_spec()->shape));
    }
    const auto num_tiles_for_output_shard = get_num_tiles_per_shard(output.shard_spec()->shape);

    log_debug(tt::LogOp, "Number of tiles per input tensor shard: {}", num_tiles_for_each_input_shard);
    log_debug(tt::LogOp, "Number of tiles for output tensor shard: {}", num_tiles_for_output_shard);

    const auto create_circular_buffer = [&program, &cores = all_cores](
                                            uint32_t index,
                                            uint32_t num_tiles,
                                            uint32_t tile_size,
                                            const tt::DataFormat& format,
                                            Buffer* buffer) -> tt::tt_metal::CBHandle {
        log_debug(
            tt::LogOp,
            "Creating CB (id={}) for {} tiles (each {} B) with total size {} B",
            index,
            num_tiles,
            tile_size,
            num_tiles * tile_size);
        tt::tt_metal::CircularBufferConfig config =
            tt::tt_metal::CircularBufferConfig(num_tiles * tile_size, {{index, format}})
                .set_page_size(index, tile_size);
        if (buffer) {
            config.set_globally_allocated_address(*buffer);
        }
        return tt::tt_metal::CreateCircularBuffer(program, cores, config);
    };

    const auto create_cb_from_tensor =
        [&create_circular_buffer](
            uint32_t idx, const Tensor& input_tensor, uint32_t total_num_tiles) -> tt::tt_metal::CBHandle {
        const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
        const auto tile_size = tt::tt_metal::detail::TileSize(data_format);
        return create_circular_buffer(idx, total_num_tiles, tile_size, data_format, input_tensor.buffer());
    };

    TT_FATAL(input_tensors.at(0).dtype() == input_tensors.at(1).dtype(), "Input tensor data types must match");
    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensors.at(0).dtype());
    const auto tile_size = tt::tt_metal::detail::TileSize(data_format);

    const uint32_t num_input_tensors = input_tensors.size();
    std::vector<CBHandle> cb_inputs(num_input_tensors);
    for (uint32_t idx = 0; idx < num_input_tensors; idx++) {
        const auto& input_tensor = input_tensors.at(idx);
        const auto total_num_tiles = get_total_num_tiles_per_shard(num_tiles_for_each_input_shard[idx]);
        cb_inputs[idx] = create_cb_from_tensor(idx, input_tensor, total_num_tiles);
    }

    const uint32_t cb_output_id = cb_inputs.size();
    const auto total_num_output_tiles = get_total_num_tiles_per_shard(num_tiles_for_output_shard);
    const CBHandle cb_output = create_cb_from_tensor(cb_output_id, output, total_num_output_tiles);

    const auto bf16_data_format = tt::tt_metal::datatype_to_dataformat_converter(DataType::BFLOAT16);
    const auto bf16_tile_size = tt::tt_metal::detail::TileSize(bf16_data_format);

    const auto in0_total_tiles_width = std::get<1>(num_tiles_for_each_input_shard[0]);
    const uint32_t cb_input0_transpose_id = cb_inputs.size() + 1;
    CBHandle cb_input0_transpose = create_circular_buffer(
        cb_input0_transpose_id, in0_total_tiles_width, bf16_tile_size, bf16_data_format, nullptr);

    const auto in1_total_tiles_width = std::get<1>(num_tiles_for_each_input_shard[1]);
    const uint32_t cb_input1_transpose_id = cb_inputs.size() + 2;
    CBHandle cb_input1_transpose = create_circular_buffer(
        cb_input1_transpose_id, in1_total_tiles_width, bf16_tile_size, bf16_data_format, nullptr);

    const auto out_total_tiles_width = in0_total_tiles_width + in1_total_tiles_width;
    const uint32_t cb_concat_id = cb_inputs.size() + 3;
    CBHandle cb_concat =
        create_circular_buffer(cb_concat_id, out_total_tiles_width, bf16_tile_size, bf16_data_format, nullptr);

    const uint32_t cb_output_transpose_id = cb_inputs.size() + 4;
    CBHandle cb_output_transpose =
        create_circular_buffer(cb_output_transpose_id, out_total_tiles_width, tile_size, data_format, nullptr);

    const bool is_rm_shard_orientation = output.shard_spec()->orientation == ShardOrientation::ROW_MAJOR;
    std::vector<uint32_t> compile_time_args_0 = {
        0,
        1,
        cb_input0_transpose_id,
        cb_input1_transpose_id,
        cb_concat_id,
        cb_output_transpose_id,
        cb_output_id,
        std::get<0>(num_tiles_for_each_input_shard[0]),
        std::get<1>(num_tiles_for_each_input_shard[0]),
        std::get<0>(num_tiles_for_each_input_shard[1]),
        std::get<1>(num_tiles_for_each_input_shard[1]),
        tile_size,
        groups,
    };
    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
        "reader_height_sharded_width_concat_two_tensors_tiled.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(compile_time_args_0));
    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
        "writer_height_sharded_width_concat_two_tensors_tiled.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(compile_time_args_0));

    // TODO: Skip the tile transpose in compute kernel if the following condition is true:
    // >> (input_tensors[0].padded_shape()[-1] / groups % TILE_WIDTH == 0
    // >> && input_tensors[1].padded_shape()[-1] / groups % TILE_WIDTH == 0)
    tt_metal::KernelHandle compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/compute/"
        "height_sharded_width_concat_two_tensors.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compile_time_args_0});

    auto override_runtime_arguments_callback = [num_input_tensors, cb_inputs, cb_output](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        for (uint32_t idx = 0; idx < num_input_tensors; idx++) {
            UpdateDynamicCircularBufferAddress(program, cb_inputs[idx], *input_tensors[idx].buffer());
        }
        UpdateDynamicCircularBufferAddress(program, cb_output, *output_tensors[0].buffer());
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

template <typename T>
static std::array<std::vector<T>, 2> split(std::vector<T> input, std::size_t index) {
    if (index > input.size()) {
        throw std::out_of_range{"split index out of range"};
    }
    std::vector<T> second{std::make_move_iterator(input.begin() + index), std::make_move_iterator(input.end())};
    input.erase(input.begin() + index, input.end());
    return {std::move(input), std::move(second)};
}

static CoreRangeSet cores_to_corerangeset(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> core_ranges;
    core_ranges.reserve(cores.size());
    for (const auto& core : cores) {
        core_ranges.push_back(CoreRange(core));
    }
    return CoreRangeSet(core_ranges);
}

tt_metal::operation::ProgramWithCallbacks s2s_rm_concat_two_tensors_height_multi_core(
    const std::vector<Tensor>& input_tensors, uint32_t dim, Tensor& output, unsigned int groups) {
    TT_FATAL(dim == 3, "Sharded concat RM only supports dim=3");
    TT_FATAL(groups == 1 || dim == 3, "Sharded concat RM only supports groups > 1 when dim=3");

    TT_FATAL(
        input_tensors.size() == 2 && input_tensors[0].padded_shape()[-1] % groups == 0 &&
            input_tensors[0].padded_shape()[-1] % groups == 0,
        "Input channels must both be evenly divisible by groups");

    tt_metal::Program program = tt_metal::CreateProgram();

    tt_metal::IDevice* device = output.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    uint32_t num_output_rows = output.padded_shape()[-2];
    uint32_t num_input_tensors = input_tensors.size();

    std::vector<CBHandle> cb_input(num_input_tensors);

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    auto all_cores = input_tensors[0].shard_spec().value().grid;

    std::vector<uint32_t> cb_ids(num_input_tensors);

    // input CBs
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        auto shard_spec = input_tensors[input_id].shard_spec().value();
        uint32_t num_input_num_units_per_shard_height = shard_spec.shape[0];
        uint32_t num_input_num_units_per_shard_width = 1;
        auto num_input_units = num_input_num_units_per_shard_height * num_input_num_units_per_shard_width;
        uint32_t input_unit_size = shard_spec.shape[1] * input_tensors[input_id].element_size();
        auto input_page_size = round_up_to_mul32(input_unit_size);
        tt_metal::CircularBufferConfig input_cb_config =
            tt_metal::CircularBufferConfig(num_input_units * input_page_size, {{input_id, cb_data_format}})
                .set_page_size(input_id, input_page_size)
                .set_globally_allocated_address(*input_tensors[input_id].buffer());
        cb_input[input_id] = tt_metal::CreateCircularBuffer(program, all_cores, input_cb_config);
        cb_ids[input_id] = input_id;
    }

    // output CB
    uint32_t cb_dst_id = 16;
    uint32_t num_output_units = output.shard_spec().value().shape[0];
    uint32_t output_unit_size = output.shard_spec().value().shape[1] * output.element_size();
    auto output_page_size = round_up_to_mul32(output_unit_size);
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(num_output_units * output_page_size, {{cb_dst_id, cb_data_format}})
            .set_page_size(cb_dst_id, output_page_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    auto output_shard_spec = output.shard_spec().value();
    uint32_t output_stick_size = output_shard_spec.shape[1] * output.element_size();

    auto input_0_shard_spec = input_tensors[0].shard_spec().value();
    auto input_1_shard_spec = input_tensors[1].shard_spec().value();
    auto input_0_stick_size = input_0_shard_spec.shape[1] * input_tensors[0].element_size();
    auto input_1_stick_size = input_1_shard_spec.shape[1] * input_tensors[1].element_size();
    auto input_0_stride = output_stick_size - input_0_stick_size;
    auto input_1_stride = output_stick_size - input_1_stick_size;
    uint32_t num_output_rows_per_core = input_tensors[0].shard_spec().value().shape[0];
    auto num_pages_per_risc = div_up(num_output_rows_per_core, 2);

    uint32_t num_output_rows_per_core_last = num_output_rows % num_output_rows_per_core;
    auto num_pages_per_risc_last = div_up(num_output_rows_per_core_last, 2);

    std::vector<uint32_t> compile_time_args_0 = {
        cb_dst_id,
        input_0_stick_size,
        input_1_stick_size,
        input_0_stride,
        input_1_stride,
        num_output_rows_per_core * num_input_tensors,
        0,
        num_pages_per_risc,
        0,
        0,
        0,
        groups};

    std::vector<uint32_t> compile_time_args_1 = {
        cb_dst_id,
        input_0_stick_size,
        input_1_stick_size,
        input_0_stride,
        input_1_stride,
        num_output_rows_per_core * num_input_tensors,
        num_pages_per_risc,
        num_output_rows_per_core,
        num_pages_per_risc * output_stick_size,
        num_pages_per_risc * input_0_stick_size,
        num_pages_per_risc * input_1_stick_size,
        groups};

    std::vector<uint32_t> compile_time_args_0_last = {
        cb_dst_id,
        input_0_stick_size,
        input_1_stick_size,
        input_0_stride,
        input_1_stride,
        num_output_rows_per_core_last * num_input_tensors,
        0,
        num_pages_per_risc_last,
        0,
        0,
        0,
        groups};

    std::vector<uint32_t> compile_time_args_1_last = {
        cb_dst_id,
        input_0_stick_size,
        input_1_stick_size,
        input_0_stride,
        input_1_stride,
        num_output_rows_per_core_last * num_input_tensors,
        num_pages_per_risc_last,
        num_output_rows_per_core_last,
        num_pages_per_risc_last * output_stick_size,
        num_pages_per_risc_last * input_0_stick_size,
        num_pages_per_risc_last * input_1_stick_size,
        groups};

    if (num_output_rows_per_core_last > 0) {
        bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
        const auto cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);
        const auto [first, last] = split(cores, cores.size() - 1);
        const auto first_cores = cores_to_corerangeset(first);
        const auto last_cores = cores_to_corerangeset(last);
        tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_height_sharded_width_concat_two_tensors.cpp",
            first_cores,
            tt_metal::ReaderDataMovementConfig(compile_time_args_0));
        tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_height_sharded_width_concat_two_tensors.cpp",
            first_cores,
            tt_metal::WriterDataMovementConfig(compile_time_args_1));

        tt_metal::KernelHandle unary_reader_kernel_last_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_height_sharded_width_concat_two_tensors.cpp",
            last_cores,
            tt_metal::ReaderDataMovementConfig(compile_time_args_0_last));
        tt_metal::KernelHandle unary_writer_kernel_last_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_height_sharded_width_concat_two_tensors.cpp",
            last_cores,
            tt_metal::WriterDataMovementConfig(compile_time_args_1_last));
    } else {
        tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_height_sharded_width_concat_two_tensors.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(compile_time_args_0));
        tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
            "reader_height_sharded_width_concat_two_tensors.cpp",
            all_cores,
            tt_metal::WriterDataMovementConfig(compile_time_args_1));
    }

    auto override_runtime_arguments_callback = [num_input_tensors, cb_input, cb_output](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
            UpdateDynamicCircularBufferAddress(program, cb_input[input_id], *input_tensors[input_id].buffer());
        }
        UpdateDynamicCircularBufferAddress(program, cb_output, *output_tensors[0].buffer());
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

// Concat sharded tensors into sharded output in row-major/tile layout. Currently it only supports height-sharded
// width concat or width-sharded height concat.
//
// It is done by copying each row of each input sharded tensor to the right offset in the sharded output tensor
// based on the sharded output width in bytes (output stride). This way works for both width and height concat.
//
// For example in width concat, rows of an input tensor are placed at the same column offset but sequential rows in
// the output. The memory address gap between neighbor input rows is exactly the output width. In height concat, all
// input rows are placed at column 0 but sequential rows in the output. The address gap between neighbor input rows
// is still the output width (which is equal to the input width).
tt_metal::operation::ProgramWithCallbacks s2s_concat_multi_core(
    const std::vector<Tensor>& input_tensors, uint32_t dim, Tensor& output) {
    TT_FATAL(dim == 2 || dim == 3, "Sharded concat only supports dim=2 or 3");
    const bool is_height_concat = dim == 2;

    tt_metal::Program program = tt_metal::CreateProgram();
    tt_metal::IDevice* device = output.device();

    const uint32_t num_input_tensors = input_tensors.size();
    const uint32_t cb_dst_id = 16;
    TT_FATAL(num_input_tensors <= cb_dst_id, "Not enough circular buffer for {} inputs.", num_input_tensors);
    const tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    const bool rm_layout = output.layout() == Layout::ROW_MAJOR;

    // Assume inputs and output have the same element size and alignment.
    const uint32_t element_size = input_tensors[0].element_size();
    const uint32_t alignment = input_tensors[0].buffer()->alignment();

    uint32_t page_size;
    uint32_t elements_per_page_width;
    uint32_t elements_per_page_height;
    if (rm_layout) {
        std::vector<uint32_t> all_stick_sizes;
        all_stick_sizes.push_back(output.shard_spec().value().shape[1]);
        std::transform(
            input_tensors.begin(), input_tensors.end(), std::back_inserter(all_stick_sizes), [](const Tensor& tensor) {
                return tensor.element_size() * tensor.shard_spec().value().shape[1];
            });
        page_size = find_greatest_common_page_size(all_stick_sizes, alignment);
        elements_per_page_width = page_size / element_size;
        elements_per_page_height = 1;
    } else {
        page_size = tt_metal::detail::TileSize(cb_data_format);
        elements_per_page_width = TILE_WIDTH;
        elements_per_page_height = TILE_HEIGHT;
    }

    std::vector<CBHandle> cb_inputs(num_input_tensors);
    std::vector<uint32_t> input_num_pages_per_stick(num_input_tensors);
    std::vector<uint32_t> input_num_sticks(num_input_tensors);
    std::vector<uint32_t> input_write_offsets(num_input_tensors);

    // Assume inputs and output have the same sharding grid.
    const auto all_cores = input_tensors[0].shard_spec().value().grid;

    // Input CBs
    uint32_t curr_input_write_offset = 0;
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        const auto shard_spec = input_tensors[input_id].shard_spec().value();
        input_num_pages_per_stick[input_id] = div_up(shard_spec.shape[1], elements_per_page_width);
        input_num_sticks[input_id] = div_up(shard_spec.shape[0], elements_per_page_height);
        input_write_offsets[input_id] = curr_input_write_offset;

        const uint32_t input_num_pages = input_num_pages_per_stick[input_id] * input_num_sticks[input_id];
        const tt_metal::CircularBufferConfig input_cb_config =
            tt_metal::CircularBufferConfig(page_size * input_num_pages, {{input_id, cb_data_format}})
                .set_page_size(input_id, page_size)
                .set_globally_allocated_address(*input_tensors[input_id].buffer());
        cb_inputs[input_id] = tt_metal::CreateCircularBuffer(program, all_cores, input_cb_config);

        curr_input_write_offset +=
            page_size * (is_height_concat ? input_num_pages : input_num_pages_per_stick[input_id]);
    }

    // Output CB
    const auto output_shard_spec = output.shard_spec().value();
    const uint32_t output_num_pages_per_stick = div_up(output_shard_spec.shape[1], elements_per_page_width);
    const uint32_t output_num_sticks = div_up(output_shard_spec.shape[0], elements_per_page_height);
    const tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(
            page_size * output_num_sticks * output_num_pages_per_stick, {{cb_dst_id, cb_data_format}})
            .set_page_size(cb_dst_id, page_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    const uint32_t output_stride = page_size * output_num_pages_per_stick;
    const std::vector<uint32_t> compile_time_args = {cb_dst_id, page_size, output_stride, num_input_tensors};

    std::vector<uint32_t> runtime_args_0;
    std::vector<uint32_t> runtime_args_1;
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        const auto input_num_sticks_per_risc = div_up(input_num_sticks[input_id], 2);
        runtime_args_0.push_back(input_num_pages_per_stick[input_id]);
        runtime_args_0.push_back(input_num_sticks_per_risc);
        runtime_args_0.push_back(input_write_offsets[input_id]);
        runtime_args_0.push_back(0);
        runtime_args_1.push_back(input_num_pages_per_stick[input_id]);
        runtime_args_1.push_back(input_num_sticks[input_id] - input_num_sticks_per_risc);
        runtime_args_1.push_back(input_write_offsets[input_id] + output_stride * input_num_sticks_per_risc);
        runtime_args_1.push_back(page_size * input_num_pages_per_stick[input_id] * input_num_sticks_per_risc);
    }

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_s2s_tensor_concat.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(compile_time_args));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_s2s_tensor_concat.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(compile_time_args));

    tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, all_cores, runtime_args_0);
    tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, all_cores, runtime_args_1);

    auto override_runtime_arguments_callback = [num_input_tensors, cb_dst_id, cb_inputs, cb_output](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
            UpdateDynamicCircularBufferAddress(program, cb_inputs[input_id], *input_tensors[input_id].buffer());
        }
        UpdateDynamicCircularBufferAddress(program, cb_output, *output_tensors[0].buffer());
    };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

tt_metal::operation::ProgramWithCallbacks s2i_rm_concat_multi_core(
    const std::vector<Tensor>& input_tensors, uint32_t dim, Tensor& output) {
    tt_metal::Program program = tt_metal::CreateProgram();

    tt_metal::IDevice* device = output.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // CoreRangeSet all_cores({CoreRange(CoreCoord(0,0), compute_with_storage_grid_size)});

    uint32_t num_output_rows = output.padded_shape()[-1];
    uint32_t num_input_tensors = input_tensors.size();

    std::vector<CBHandle> cb_input(num_input_tensors);
    std::vector<uint32_t> input_num_units_per_shard_height(num_input_tensors);
    std::vector<uint32_t> input_num_units_per_shard_width(num_input_tensors);

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    auto all_cores = input_tensors[0].shard_spec().value().grid;

    std::vector<uint32_t> cb_ids(num_input_tensors);
    uint32_t input_unit_size = input_tensors[0].shard_spec().value().shape[1] * input_tensors[0].element_size();
    // input CBs
    for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
        auto shard_spec = input_tensors[input_id].shard_spec().value();
        input_num_units_per_shard_height[input_id] = shard_spec.shape[0];
        input_num_units_per_shard_width[input_id] = 1;
        auto num_input_units = input_num_units_per_shard_height[input_id] * input_num_units_per_shard_width[input_id];
        auto input_page_size = round_up_to_mul32(input_unit_size);
        tt_metal::CircularBufferConfig input_cb_config =
            tt_metal::CircularBufferConfig(num_input_units * input_page_size, {{input_id, cb_data_format}})
                .set_page_size(input_id, input_page_size)
                .set_globally_allocated_address(*input_tensors[input_id].buffer());
        cb_input[input_id] = tt_metal::CreateCircularBuffer(program, all_cores, input_cb_config);
        cb_ids[input_id] = input_id;
    }

    bool dst_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {num_input_tensors};
    std::vector<uint32_t> writer_compile_time_args = {num_input_tensors, std::uint32_t(dst_is_dram)};

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_s2i_width.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/writer_s2i_width.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    bool row_wise = input_tensors[0].shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
    auto cores = corerange_to_cores(all_cores, std::nullopt, row_wise);
    auto input_cores = input_tensors[0].shard_spec().value().grid;
    uint32_t num_output_rows_per_core = div_up(num_output_rows, input_cores.num_cores());

    uint32_t core_id = 0;
    for (auto core : cores) {
        auto input_shard_spec = input_tensors[0].shard_spec().value();
        uint32_t curr_num_output_rows;
        if (input_cores.contains(core)) {
            curr_num_output_rows = num_output_rows_per_core;
        } else {
            curr_num_output_rows = 0;
        }

        std::vector<uint32_t> reader_runtime_args = {};
        std::vector<uint32_t> writer_runtime_args = {
            output.buffer()->address(),
            core_id,
            curr_num_output_rows,
            input_unit_size,
            num_input_tensors * input_shard_spec.shape[0],
            input_shard_spec.shape[0]};
        for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
            reader_runtime_args.push_back(input_id);
            reader_runtime_args.push_back(input_shard_spec.shape[0]);
            writer_runtime_args.push_back(input_id);
        }
        tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);

        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
        core_id++;
    }

    auto override_runtime_arguments_callback =
        [unary_reader_kernel_id, unary_writer_kernel_id, all_cores, num_input_tensors](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            bool row_wise = input_tensors[0].shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
            auto dst_buffer = output_tensors.at(0).buffer();
            auto cores = corerange_to_cores(all_cores, std::nullopt, row_wise);
            auto input_cores = input_tensors[0].shard_spec().value().grid;
            uint32_t num_output_rows = output_tensors[0].padded_shape()[-1];
            uint32_t num_output_rows_per_core = div_up(num_output_rows, input_cores.num_cores());
            for (auto core : cores) {
                uint32_t curr_num_input_tensors;
                uint32_t curr_num_output_rows;
                if (input_cores.contains(core)) {
                    curr_num_input_tensors = num_input_tensors;
                    curr_num_output_rows = num_output_rows_per_core;
                } else {
                    curr_num_input_tensors = 0;
                    curr_num_output_rows = 0;
                }

                std::vector<uint32_t> reader_runtime_args = {curr_num_input_tensors};
                std::vector<uint32_t> writer_runtime_args = {
                    dst_buffer->address(), curr_num_input_tensors, curr_num_output_rows};
                for (uint32_t input_id = 0; input_id < num_input_tensors; input_id++) {
                    UpdateDynamicCircularBufferAddress(program, input_id, *dst_buffer);
                    auto input_shard_spec = input_tensors[input_id].shard_spec().value();
                    reader_runtime_args.push_back(input_id);
                    reader_runtime_args.push_back(input_shard_spec.shape[1]);
                    writer_runtime_args.push_back(input_id);
                }
                tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);

                tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

tt_metal::operation::ProgramWithCallbacks sharded_concat_multi_core(
    const std::vector<Tensor>& input_tensors, uint32_t dim, Tensor& output, unsigned int groups) {
    if (output.is_sharded()) {
        if (input_tensors.size() == 2) {
            // There are unrolled kernels for the case where we have 2 hight-sharded kernels. Currently
            // we only support 2 tile inputs OR 2 row-major inputs.
            TT_FATAL(
                input_tensors.at(0).layout() == input_tensors.at(1).layout(),
                "Expected all input tensors to have the same layout");
            if (input_tensors.at(0).layout() == Layout::ROW_MAJOR) {
                return s2s_rm_concat_two_tensors_height_multi_core(input_tensors, dim, output, groups);
            } else {
                return s2s_tiled_concat_two_tensors_height_multi_core(input_tensors, dim, output, groups);
            }
        } else {
            TT_FATAL(
                groups == 1,
                "Sharded ttnn.concat with groups > 1 is only supported for 2 sharded input and sharded output "
                "tensors");
            return s2s_concat_multi_core(input_tensors, dim, output);
        }
    } else {
        TT_FATAL(
            groups == 1,
            "Sharded ttnn.concat with groups > 1 is only supported for 2 sharded input and sharded output tensors");
        return s2i_rm_concat_multi_core(input_tensors, dim, output);
    }
}

tt_metal::operation::ProgramWithCallbacks concat_multi_core(
    const std::vector<Tensor>& input_tensors, const uint32_t dim, const Tensor& output) {
    tt_metal::Program program = tt_metal::CreateProgram();

    tt_metal::IDevice* device = output.device();

    const tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    const bool rm_layout = output.layout() == Layout::ROW_MAJOR;

    constexpr bool rm_orientation = false;

    uint32_t num_output_pages;
    uint32_t single_page_size;
    uint32_t common_align_len = std::max(input_tensors[0].buffer()->alignment(), output.buffer()->alignment());
    if (rm_layout) {
        num_output_pages = output.physical_volume() / output.padded_shape()[-1];
        single_page_size = tt::align(output.element_size() * output.padded_shape()[-1], common_align_len);
    } else {
        num_output_pages = output.physical_volume() / TILE_HW;
        single_page_size = tt_metal::detail::TileSize(cb_data_format);
    }

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_pages, rm_orientation);

    uint32_t num_input_tensors = input_tensors.size();

    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_pages = 2;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_pages * single_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_page_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t num_dims = output.padded_shape().rank();

    std::vector<uint32_t> src_addr(num_input_tensors);
    std::vector<bool> is_dram(num_input_tensors);
    std::vector<uint32_t> num_pages_per_block(num_input_tensors);
    std::vector<uint32_t> page_id_per_tensor(num_input_tensors);
    // Only used for RM
    std::vector<uint32_t> page_size_per_tensor(num_input_tensors);

    uint32_t num_accum_pages = 1;
    uint32_t scale_factor = 1;

    // RM is special cased in the loop (dim_units = 1 for last dim else it's the dim size)
    if (!rm_layout) {
        if (dim == num_dims - 2) {
            scale_factor = TILE_HEIGHT;
        } else if (dim == num_dims - 1) {
            scale_factor = TILE_WIDTH;
        }
    }

    for (uint32_t i = dim + 1; i < num_dims; ++i) {
        num_accum_pages *= output.padded_shape()[i];
    }
    if (rm_layout) {
        if (num_dims > 1 && dim < num_dims - 1) {
            num_accum_pages /= output.padded_shape()[-1];
        }
    } else {
        if (dim < num_dims - 2) {
            num_accum_pages /= TILE_HW;
        } else if (dim == num_dims - 2) {
            num_accum_pages /= TILE_WIDTH;
        }
    }

    uint32_t num_output_pages_per_block = 0;

    if (rm_layout) {
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            auto buffer = input_tensors[i].buffer();
            src_addr[i] = buffer->address();
            is_dram[i] = buffer->buffer_type() == tt_metal::BufferType::DRAM;
            page_size_per_tensor[i] = buffer->page_size();
            if (dim == num_dims - 1) {
                num_pages_per_block[i] = num_accum_pages;
            } else {
                uint32_t dim_pages = input_tensors[i].padded_shape()[dim];
                num_pages_per_block[i] = num_accum_pages * dim_pages;
                num_output_pages_per_block += num_accum_pages * dim_pages;
            }
        }
        if (dim == num_dims - 1) {
            num_output_pages_per_block = 1;
        }
    } else {
        for (uint32_t i = 0; i < num_input_tensors; ++i) {
            auto buffer = input_tensors[i].buffer();
            src_addr[i] = buffer->address();
            is_dram[i] = buffer->buffer_type() == tt_metal::BufferType::DRAM;
            uint32_t dim_pages = input_tensors[i].padded_shape()[dim] / scale_factor;
            num_pages_per_block[i] = num_accum_pages * dim_pages;
            num_output_pages_per_block += num_accum_pages * dim_pages;
        }
    }
    std::vector<uint32_t> common_reader_kernel_args = {0, 0, 0};
    common_reader_kernel_args.insert(common_reader_kernel_args.end(), src_addr.begin(), src_addr.end());
    common_reader_kernel_args.insert(common_reader_kernel_args.end(), is_dram.begin(), is_dram.end());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_pages_per_block.begin(), num_pages_per_block.end());
    if (rm_layout) {
        common_reader_kernel_args.insert(
            common_reader_kernel_args.end(), page_size_per_tensor.begin(), page_size_per_tensor.end());
    }

    // Reader compile-time args
    // Data is 32 byte aligned
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)num_input_tensors,
    };

    std::map<std::string, std::string> concat_defines;

    if (rm_layout && dim == num_dims - 1) {
        concat_defines["WIDTH_CONCAT"] = "1";
    }

    std::vector<uint32_t> writer_compile_time_args = {// interleaved accessor args
                                                      (std::uint32_t)src0_cb_index,
                                                      (std::uint32_t)dst_is_dram,
                                                      0,
                                                      0};

    // Tilized reader
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        rm_layout ? "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
                    "reader_concat_stick_layout_interleaved_start_id.cpp"
                  : "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
                    "reader_concat_interleaved_start_id.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, concat_defines));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        rm_layout
            ? "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp"
            : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
              "writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, rm_orientation);
    uint32_t g1_num_cores = core_group_1.num_cores();
    for (uint32_t i = 0, num_pages_written = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        uint32_t num_pages_per_core = 0;
        if (i < g1_num_cores) {
            num_pages_per_core = num_tiles_per_core_group_1;
        } else {
            num_pages_per_core = num_tiles_per_core_group_2;
        }
        uint32_t block_id = num_pages_written / num_output_pages_per_block;
        uint32_t id_within_block = num_pages_written % num_output_pages_per_block;
        uint32_t curr_tensor = 0;
        uint32_t curr_tensor_id = 0;
        for (uint32_t j = 0; j < num_input_tensors; j++) {
            page_id_per_tensor[j] = block_id * num_pages_per_block[j];
            if (id_within_block == 0) {
                continue;
            } else if (id_within_block >= num_pages_per_block[j]) {
                page_id_per_tensor[j] += num_pages_per_block[j];
                id_within_block -= num_pages_per_block[j];
                curr_tensor = j + 1;
            } else {
                page_id_per_tensor[j] += id_within_block;
                curr_tensor = j;
                curr_tensor_id = id_within_block;
                id_within_block = 0;
            }
        }

        std::vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        reader_kernel_args[0] = num_pages_per_core;
        reader_kernel_args[1] = curr_tensor;
        reader_kernel_args[2] = curr_tensor_id;
        reader_kernel_args.insert(reader_kernel_args.end(), page_id_per_tensor.begin(), page_id_per_tensor.end());

        std::vector<uint32_t> writer_kernel_args;
        if (rm_layout) {
            writer_kernel_args = {
                dst_buffer->address(), output.buffer()->page_size(), num_pages_per_core, num_pages_written};
        } else {
            writer_kernel_args = {dst_buffer->address(), num_pages_per_core, num_pages_written};
        }
        tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);

        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);
        num_pages_written += num_pages_per_core;
    }

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id, cores](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        std::vector<uint32_t> src_addrs(input_tensors.size());
        for (uint32_t i = 0; i < input_tensors.size(); ++i) {
            src_addrs[i] = input_tensors[i].buffer()->address();
        }

        auto dst_buffer = output_tensors.at(0).buffer();

        for (const auto& core : cores) {
            {
                auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                std::copy(src_addrs.begin(), src_addrs.end(), runtime_args.data() + 3);
            }

            {
                auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::data_movement::detail
