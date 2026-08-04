// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "nlp_concat_heads_decode_program_factory.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

ProgramDescriptor NLPConcatHeadsDecodeProgramFactory::create_descriptor(
    const NlpConcatHeadsDecodeParams& /*operation_attributes*/,
    const NlpConcatHeadsDecodeInputs& tensor_args,
    Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    ProgramDescriptor desc;

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t head_dim = input_shape[-1];
    const uint32_t batch = input_shape[1];

    tt_metal::IDevice* device = input_tensor.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    uint32_t element_size = input_tensor.element_size();
    uint32_t sub_tile_line_bytes = 16 * element_size;
    auto q_shard_spec = output.shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;
    auto in_shard_spec = input_tensor.shard_spec().value();
    auto in_cores = in_shard_spec.grid;

    uint32_t q_output_cb_index = CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = q_num_tiles * single_tile_size,
        .core_ranges = q_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(q_output_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = output.buffer(),
    });

    Buffer* in_buffer = input_tensor.buffer();

    // cores to read and write to output
    uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    auto core_grid = q_cores.bounding_box();
    uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    // cores for input
    auto in_core_grid = in_cores.bounding_box();
    uint32_t in_num_cores_x = in_core_grid.end_coord.x + 1, in_num_cores_y = in_core_grid.end_coord.y + 1;

    std::vector<uint32_t> noc_x_coords;
    noc_x_coords.reserve(in_num_cores_x);
    for (uint32_t x = 0; x < in_num_cores_x; ++x) {
        noc_x_coords.push_back(device->worker_core_from_logical_core({x, 0}).x);
    }
    std::vector<uint32_t> noc_y_coords;
    noc_y_coords.reserve(in_num_cores_y);
    for (uint32_t y = 0; y < in_num_cores_y; ++y) {
        noc_y_coords.push_back(device->worker_core_from_logical_core({0, y}).y);
    }

    // We parallelize the reader on risc0 and risc1, where each risc reads a sub-tile of the input (phase1 and phase2 of
    // a tile respectively)
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)element_size,
        (std::uint32_t)sub_tile_line_bytes,
        q_output_cb_index,
        head_size,
        batch,
        head_tiles,
        1,  // read the first phase
        in_num_cores_x,
        in_num_cores_y};

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_concat_heads_decode.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = q_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_time_args = reader_compile_time_args;
    writer_compile_time_args[6] = 2;  // read the second phase

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_concat_heads_decode.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = q_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    for (uint32_t i = 0; i < num_cores; ++i) {
        // Each output core i corresponds to head index i. Within the input shard, that head lives in
        // head-tile (i / 32) at row (i % 32). The two cases below pick the row's byte offset within
        // a single 32x32 tile (face 0 for rows < 16, face 2 for rows >= 16); add the head-tile skip
        // to land in the right tile when padded_heads > 32.
        uint32_t head_tile_idx = i / 32;
        uint32_t head_in_tile = i % 32;
        uint32_t in_tile_offset_by_batch =
            (head_in_tile < 16 ? head_in_tile * sub_tile_line_bytes
                               : (head_in_tile - 16) * sub_tile_line_bytes + 512 * element_size) +
            head_tile_idx * head_size;

        const auto& core = cores[i];
        KernelDescriptor::RTArgList rt_args;
        rt_args.reserve(2 + in_num_cores_x + in_num_cores_y);
        rt_args.push_back(in_tile_offset_by_batch);
        rt_args.push_back(in_buffer);
        rt_args.append(noc_x_coords);
        rt_args.append(noc_y_coords);

        reader_desc.emplace_runtime_args(core, rt_args);
        writer_desc.emplace_runtime_args(core, rt_args);
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
