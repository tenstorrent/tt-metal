// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/split/device/split_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;

namespace {

// Assigns runtime args for the TILE N-way split.
// Each core belongs to exactly one chunk group and writes to one output buffer.
//
// Core layout (row-major in metal notation):
//   rows  (x): z_batch * num_cores_x  — parallelizes Z (dim 1) and dim-2 tiles
//   cols  (y): num_chunks * num_cores_per_chunk — parallelizes dim-3 tiles, grouped by chunk
//
// Within a column group k (k=0..num_chunks-1):
//   cores in that group write tiles from [k * tiles_per_chunk, (k+1) * tiles_per_chunk) of the last dim
//   to output buffer k.
void setup_runtime(
    KernelDescriptor& reader_desc,
    KernelDescriptor& writer_desc,
    const uint32_t num_chunks,
    const uint32_t num_cores_per_chunk,
    const uint32_t num_cores_z,
    const uint32_t num_cores_x,
    const uint32_t per_core_tiles_y,
    const uint32_t per_core_tiles_x,
    const uint32_t num_tiles_per_z,
    Buffer* in0_buffer,
    const std::vector<Buffer*>& output_buffers) {
    // Total Y-cores = num_chunks * num_cores_per_chunk
    const uint32_t num_cores_c = num_chunks * num_cores_per_chunk;

    for (uint32_t id_r_outer = 0; id_r_outer < num_cores_z; id_r_outer++) {
        for (uint32_t id_r_inner = 0; id_r_inner < num_cores_x; id_r_inner++) {
            uint32_t id_r = id_r_outer * num_cores_x + id_r_inner;

            // Starting tile ID in the INPUT buffer for this (z, x) row of cores.
            uint32_t id_r_reader =
                (id_r_outer * num_tiles_per_z) + (id_r_inner * per_core_tiles_y * num_cores_c * per_core_tiles_x);

            // Corresponding starting tile in each OUTPUT buffer (output has 1/num_chunks fewer Y tiles).
            uint32_t id_r_writer = id_r_reader / num_chunks;

            for (uint32_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
                for (uint32_t id_c_inner = 0; id_c_inner < num_cores_per_chunk; id_c_inner++) {
                    uint32_t id_c = chunk_id * num_cores_per_chunk + id_c_inner;
                    CoreCoord core = {(std::size_t)id_r, (std::size_t)id_c};

                    uint32_t reader_core_id = id_c * per_core_tiles_y + id_r_reader;
                    uint32_t writer_core_id = id_c_inner * per_core_tiles_y + id_r_writer;

                    reader_desc.emplace_runtime_args(core, {reader_core_id, in0_buffer, (std::uint32_t)0});

                    writer_desc.emplace_runtime_args(core, {writer_core_id, output_buffers[chunk_id]});
                }
            }
        }
    }
}

}  // namespace

ProgramDescriptor SplitProgramFactory::create_descriptor(
    const SplitParams& operation_attributes, const SplitInputs& tensor_args, std::vector<Tensor>& tensor_return_value) {
    const auto& input_tensor = tensor_args.input;
    const uint32_t num_chunks = static_cast<uint32_t>(operation_attributes.num_splits);

    auto input_shape = input_tensor.padded_shape();
    IDevice* device = input_tensor.device();
    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    Buffer* in0_buffer = input_tensor.buffer();

    // Collect output buffers and validate they are all the same type / page size.
    TT_FATAL(
        tensor_return_value.size() == num_chunks,
        "Number of output tensors ({}) must equal number of chunks ({})",
        tensor_return_value.size(),
        num_chunks);
    std::vector<Buffer*> output_buffers;
    output_buffers.reserve(num_chunks);
    for (uint32_t i = 0; i < num_chunks; i++) {
        Buffer* buf = tensor_return_value[i].buffer();
        TT_FATAL(buf != nullptr, "Output {} buffer should be allocated on device!", i);
        if (i > 0) {
            TT_FATAL(
                buf->buffer_type() == output_buffers[0]->buffer_type(),
                "All output buffers must have the same buffer type");
            TT_FATAL(
                buf->aligned_page_size() == output_buffers[0]->aligned_page_size(),
                "All output buffers must have the same aligned page size");
        }
        output_buffers.push_back(buf);
    }

    uint32_t z = input_shape[1];
    uint32_t num_tiles_dim_2 = input_shape[2] / tt::constants::TILE_HEIGHT;
    uint32_t num_tiles_dim_3 = input_shape[3] / tt::constants::TILE_WIDTH;
    uint32_t num_cores_x_limit = device->compute_with_storage_grid_size().x;
    uint32_t num_cores_y_limit = device->compute_with_storage_grid_size().y;

    // Parallelize the Z (dim 1) dimension across separate core rows.
    uint32_t num_cores_z = z;

    // Parallelize dim-2 (height tiles) across X cores.
    auto [num_cores_x, per_core_tiles_x] =
        get_max_cores_divisible_by_tiles_per_core_tiles(num_tiles_dim_2, num_cores_x_limit / num_cores_z);

    // Parallelize dim-3 (width tiles) across Y cores, grouped by chunk.
    // We need num_cores_y to be a multiple of num_chunks so each chunk gets an equal group.
    uint32_t tiles_per_chunk = num_tiles_dim_3 / num_chunks;
    uint32_t max_cores_per_chunk = num_cores_y_limit / num_chunks;
    auto [num_cores_per_chunk, per_core_tiles_y] =
        get_max_cores_divisible_by_tiles_per_core_tiles(tiles_per_chunk, max_cores_per_chunk);

    uint32_t num_cores_c = num_cores_per_chunk * num_chunks;  // total Y-cores
    uint32_t num_cores_r = num_cores_x * num_cores_z;         // total X-cores (rows)

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_r - 1, (std::size_t)start_core_y + num_cores_c - 1});

    uint32_t num_tiles_per_z = (per_core_tiles_x * num_cores_x) * (per_core_tiles_y * num_cores_c);
    uint32_t z_stride_read = num_tiles_per_z;
    uint32_t y_stride_read = per_core_tiles_y * num_cores_c;

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)(z / num_cores_z),
        (std::uint32_t)per_core_tiles_x,
        (std::uint32_t)per_core_tiles_y,
        (std::uint32_t)z_stride_read,
        (std::uint32_t)y_stride_read};
    TensorAccessorArgs(*in0_buffer).append_to(reader_compile_time_args);

    uint32_t z_stride_write = num_tiles_per_z / num_chunks;
    uint32_t y_stride_write = per_core_tiles_y * num_cores_per_chunk;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)per_core_tiles_x,
        (std::uint32_t)per_core_tiles_y,
        (std::uint32_t)(z / num_cores_z),
        (std::uint32_t)z_stride_write,
        (std::uint32_t)y_stride_write};
    // ONE shared TensorAccessorArgs: all output chunks have the same buffer type.
    TensorAccessorArgs(*output_buffers[0]).append_to(writer_compile_time_args);

    ProgramDescriptor desc;

    constexpr uint32_t src0_cb_index = 0;
    constexpr uint32_t num_input_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = CoreRangeSet{all_cores},
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/split/device/kernels/dataflow/"
        "reader_tm_tile_layout_split_two_chunks.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = CoreRangeSet{all_cores};
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/split/device/kernels/dataflow/"
        "writer_split_n_chunks_tile.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = CoreRangeSet{all_cores};
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    setup_runtime(
        reader_desc,
        writer_desc,
        num_chunks,
        num_cores_per_chunk,
        num_cores_z,
        num_cores_x,
        per_core_tiles_y,
        per_core_tiles_x,
        num_tiles_per_z,
        in0_buffer,
        output_buffers);

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
