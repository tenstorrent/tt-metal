// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_s2s_tiled_program_factory.hpp"

#include <algorithm>

#include "tt-metalium/buffer.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor ConcatS2STiledProgramFactory::create_descriptor(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    const std::vector<Tensor>& input_tensors = tensor_args.input_tensors;
    const unsigned int groups = operation_attributes.groups;
    Tensor& output = tensor_return_value;

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

    ProgramDescriptor desc;
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

    TT_FATAL(input_tensors.at(0).dtype() == input_tensors.at(1).dtype(), "Input tensor data types must match");
    const tt::DataFormat data_format = datatype_to_dataformat_converter(input_tensors.at(0).dtype());
    const uint32_t tile_size = tt::tile_size(data_format);
    const uint32_t num_input_tensors = input_tensors.size();

    for (uint32_t idx = 0; idx < num_input_tensors; idx++) {
        const Tensor& input_tensor = input_tensors.at(idx);
        const uint32_t total_num_tiles = get_total_num_tiles_per_shard(num_tiles_for_each_input_shard[idx]);
        desc.cbs.push_back(CBDescriptor{
            .total_size = total_num_tiles * tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx),
                .data_format = datatype_to_dataformat_converter(input_tensor.dtype()),
                .page_size = tt::tile_size(datatype_to_dataformat_converter(input_tensor.dtype())),
            }}},
            .buffer = input_tensor.buffer(),
        });
    }

    const uint32_t cb_output_id = num_input_tensors;
    const uint32_t total_num_output_tiles = get_total_num_tiles_per_shard(num_tiles_for_output_shard);
    desc.cbs.push_back(CBDescriptor{
        .total_size = total_num_output_tiles * tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_output_id),
            .data_format = datatype_to_dataformat_converter(output.dtype()),
            .page_size = tt::tile_size(datatype_to_dataformat_converter(output.dtype())),
        }}},
        .buffer = output.buffer(),
    });

    tt::DataFormat cb_data_format = data_format;
    uint32_t cb_tile_size = tile_size;
    const bool is_bf8 = input_tensors[0].dtype() == DataType::BFLOAT8_B;
    if (is_bf8) {
        cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(DataType::BFLOAT16);
        cb_tile_size = tt::tile_size(cb_data_format);
    }

    const uint32_t in0_total_tiles_width = num_tiles_for_each_input_shard[0].second;
    const uint32_t cb_input0_transpose_id = num_input_tensors + 1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_total_tiles_width * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_input0_transpose_id),
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });

    const uint32_t in1_total_tiles_width = num_tiles_for_each_input_shard[1].second;
    const uint32_t cb_input1_transpose_id = num_input_tensors + 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_total_tiles_width * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_input1_transpose_id),
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });

    const uint32_t out_total_tiles_width = in0_total_tiles_width + in1_total_tiles_width;
    const uint32_t cb_concat_id = num_input_tensors + 3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_total_tiles_width * cb_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_concat_id),
            .data_format = cb_data_format,
            .page_size = cb_tile_size,
        }}},
    });

    const uint32_t cb_output_transpose_id = num_input_tensors + 4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_total_tiles_width * tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_output_transpose_id),
            .data_format = data_format,
            .page_size = tile_size,
        }}},
    });

    // TODO: Skip the tile transpose in compute kernel if the following condition is true:
    // >> (input_tensors[0].padded_shape()[-1] / groups % TILE_WIDTH == 0
    // >> && input_tensors[1].padded_shape()[-1] / groups % TILE_WIDTH == 0)
    constexpr uint32_t MAX_1_BYTE_TILES_PER_BATCH = 16;
    const uint32_t batch_size = MAX_1_BYTE_TILES_PER_BATCH / input_tensors[0].element_size();

    // Calculate stride sizes to determine if we can use single-packet NOC reads
    // For BF8, the kernel uses bf16_tile_size (2048 bytes) for stride calculation
    const uint32_t stride_tile_size = is_bf8 ? cb_tile_size : tile_size;
    const uint32_t input0_stride = stride_tile_size * num_tiles_for_each_input_shard[0].second / groups;
    const uint32_t input1_stride = stride_tile_size * num_tiles_for_each_input_shard[1].second / groups;

    // NOC_MAX_BURST_SIZE from noc_parameters.h: Wormhole = 8192, Blackhole = 16384
    const uint32_t noc_max_burst_size = (input_tensors[0].device()->arch() == tt::ARCH::BLACKHOLE) ? 16384 : 8192;
    const bool use_single_packet_read = (input0_stride <= noc_max_burst_size && input1_stride <= noc_max_burst_size);

    KernelDescriptor::CompileTimeArgs compile_time_args = {
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

    KernelDescriptor::Defines reader_defines;
    if (is_bf8) {
        reader_defines.emplace_back("BF8", "1");
    }
    if (use_single_packet_read) {
        reader_defines.emplace_back("USE_SINGLE_PACKET_READ", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
        "reader_height_sharded_width_concat_two_tensors_tiled.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = compile_time_args;
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
        "writer_height_sharded_width_concat_two_tensors_tiled.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/compute/"
        "height_sharded_width_concat_two_tensors.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compile_time_args);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = data_format == tt::DataFormat::Float32 || data_format == tt::DataFormat::Int32 ||
                            data_format == tt::DataFormat::UInt32,
        .math_approx_mode = false,
    };

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::prim
