// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_s2s_tiled_program_factory.hpp"

#include <algorithm>

#include "tt-metalium/buffer.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>

namespace ttnn::prim {

ConcatS2STiledProgramFactory::cached_program_t ConcatS2STiledProgramFactory::create(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const std::vector<Tensor>& input_tensors = tensor_args.input_tensors;
    const uint32_t dim = operation_attributes.dim;
    const unsigned int groups = operation_attributes.groups;
    Tensor& output = tensor_return_value;
    // If we end up here for concat with more than 2 tensors on any other dim we should have
    // taken another path
    TT_FATAL(dim == 3, "Sharded concat with tiled inputs only supports dim=3 (was {})", dim);
    TT_FATAL(input_tensors.size() == 2, "Expected 2 input tensors (was {})", input_tensors.size());
    TT_FATAL(input_tensors[0].shard_spec().has_value(), "Input tensor 0 must be sharded");
    TT_FATAL(input_tensors[1].shard_spec().has_value(), "Input tensor 1 must be sharded");
    TT_FATAL(output.shard_spec().has_value(), "Output must be sharded");

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

    Program program = CreateProgram();

    const CoreRangeSet all_cores = input_tensors[0].shard_spec().value().grid;  // assume all inputs have same grid

    const auto get_num_tiles_per_shard = [](const ShardSpec& shard_spec) -> std::pair<uint32_t, uint32_t> {
        const std::array<uint32_t, 2> shard_shape = shard_spec.shape;
        TT_FATAL(shard_shape[0] % TILE_HEIGHT == 0, "Shard height must be aligned to tile height");
        TT_FATAL(shard_shape[1] % TILE_WIDTH == 0, "Shard width must be aligned to tile width");
        const uint32_t num_tiles_along_height = shard_shape[0] / TILE_HEIGHT;
        const uint32_t num_tiles_along_width = shard_shape[1] / TILE_WIDTH;
        TT_FATAL(num_tiles_along_height != 0 && num_tiles_along_width != 0, "Expected tensor to have at least 1 tiles");
        return {num_tiles_along_height, num_tiles_along_width};
    };

    const auto get_total_num_tiles_per_shard = [](const std::pair<uint32_t, uint32_t>& num_tiles) -> uint32_t {
        return num_tiles.first * num_tiles.second;
    };

    std::vector<std::pair<uint32_t, uint32_t>> num_tiles_for_each_input_shard;
    num_tiles_for_each_input_shard.reserve(input_tensors.size());
    std::transform(
        input_tensors.begin(),
        input_tensors.end(),
        std::back_inserter(num_tiles_for_each_input_shard),
        [&get_num_tiles_per_shard](const Tensor& input_tensor) {
            return get_num_tiles_per_shard(input_tensor.shard_spec().value());
        });
    const std::pair<uint32_t, uint32_t> num_tiles_for_output_shard =
        get_num_tiles_per_shard(output.shard_spec().value());

    const auto create_circular_buffer = [&program, &cores = all_cores](
                                            uint32_t index,
                                            uint32_t num_tiles,
                                            uint32_t tile_size,
                                            const tt::DataFormat& format,
                                            Buffer* buffer) -> CBHandle {
        CircularBufferConfig config =
            CircularBufferConfig(num_tiles * tile_size, {{index, format}}).set_page_size(index, tile_size);
        if (buffer) {
            config.set_globally_allocated_address(*buffer);
        }
        return CreateCircularBuffer(program, cores, config);
    };

    const auto create_cb_from_tensor =
        [&create_circular_buffer](uint32_t idx, const Tensor& input_tensor, uint32_t total_num_tiles) -> CBHandle {
        const auto data_format = datatype_to_dataformat_converter(input_tensor.dtype());
        const auto tile_size = tt::tile_size(data_format);
        return create_circular_buffer(idx, total_num_tiles, tile_size, data_format, input_tensor.buffer());
    };

    TT_FATAL(input_tensors.at(0).dtype() == input_tensors.at(1).dtype(), "Input tensor data types must match");
    const tt::DataFormat data_format = datatype_to_dataformat_converter(input_tensors.at(0).dtype());
    const uint32_t tile_size = tt::tile_size(data_format);

    const uint32_t num_input_tensors = input_tensors.size();
    std::vector<CBHandle> cb_inputs(num_input_tensors);
    for (uint32_t idx = 0; idx < num_input_tensors; idx++) {
        const Tensor& input_tensor = input_tensors.at(idx);
        const uint32_t total_num_tiles = get_total_num_tiles_per_shard(num_tiles_for_each_input_shard[idx]);
        cb_inputs[idx] = create_cb_from_tensor(idx, input_tensor, total_num_tiles);
    }

    const uint32_t cb_output_id = cb_inputs.size();
    const uint32_t total_num_output_tiles = get_total_num_tiles_per_shard(num_tiles_for_output_shard);
    const CBHandle cb_output = create_cb_from_tensor(cb_output_id, output, total_num_output_tiles);

    tt::DataFormat cb_data_format = data_format;
    uint32_t cb_tile_size = tile_size;
    const bool is_bf8 = input_tensors[0].dtype() == DataType::BFLOAT8_B;
    if (is_bf8) {
        cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(DataType::BFLOAT16);
        cb_tile_size = tt::tile_size(cb_data_format);
    }

    const uint32_t in0_total_tiles_width = num_tiles_for_each_input_shard[0].second;
    const uint32_t cb_input0_transpose_id = cb_inputs.size() + 1;
    create_circular_buffer(cb_input0_transpose_id, in0_total_tiles_width, cb_tile_size, cb_data_format, nullptr);

    const uint32_t in1_total_tiles_width = num_tiles_for_each_input_shard[1].second;
    const uint32_t cb_input1_transpose_id = cb_inputs.size() + 2;
    create_circular_buffer(cb_input1_transpose_id, in1_total_tiles_width, cb_tile_size, cb_data_format, nullptr);

    const uint32_t out_total_tiles_width = in0_total_tiles_width + in1_total_tiles_width;
    const uint32_t cb_concat_id = cb_inputs.size() + 3;
    create_circular_buffer(cb_concat_id, out_total_tiles_width, cb_tile_size, cb_data_format, nullptr);

    const uint32_t cb_output_transpose_id = cb_inputs.size() + 4;
    create_circular_buffer(cb_output_transpose_id, out_total_tiles_width, tile_size, data_format, nullptr);

    // TODO: Skip the tile transpose in compute kernel if the following condition is true:
    // >> (input_tensors[0].padded_shape()[-1] / groups % TILE_WIDTH == 0
    // >> && input_tensors[1].padded_shape()[-1] / groups % TILE_WIDTH == 0)
    constexpr uint32_t MAX_1_BYTE_TILES_PER_BATCH = 16;
    const uint32_t batch_size = MAX_1_BYTE_TILES_PER_BATCH / input_tensors[0].element_size();

    std::vector<uint32_t> compile_time_args_0 = {
        0,
        1,
        cb_input0_transpose_id,
        cb_input1_transpose_id,
        cb_concat_id,
        cb_output_transpose_id,
        cb_output_id,
        num_tiles_for_each_input_shard[0].first,
        num_tiles_for_each_input_shard[0].second,
        num_tiles_for_each_input_shard[1].first,
        num_tiles_for_each_input_shard[1].second,
        tile_size,
        groups,
        batch_size,
    };
    std::map<std::string, std::string> reader_defines;
    if (is_bf8) {
        reader_defines["BF8"] = "1";
    }
    CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
        "reader_height_sharded_width_concat_two_tensors_tiled.cpp",
        all_cores,
        ReaderDataMovementConfig(compile_time_args_0, std::move(reader_defines)));
    CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
        "writer_height_sharded_width_concat_two_tensors_tiled.cpp",
        all_cores,
        WriterDataMovementConfig(compile_time_args_0));

    CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/compute/"
        "height_sharded_width_concat_two_tensors.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compile_time_args_0});

    return {
        std::move(program),
        {.num_input_tensors = num_input_tensors,
         .cb_inputs = cb_inputs,
         .cb_output = cb_output,
         .all_cores = all_cores}};
}

void ConcatS2STiledProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ConcatParams& /*operation_attributes*/,
    const ConcatInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;

    for (uint32_t input_id = 0; input_id < shared_vars.num_input_tensors; input_id++) {
        UpdateDynamicCircularBufferAddress(
            program, shared_vars.cb_inputs[input_id], *tensor_args.input_tensors[input_id].buffer());
    }
    UpdateDynamicCircularBufferAddress(program, shared_vars.cb_output, *tensor_return_value.buffer());
}

}  // namespace ttnn::prim
