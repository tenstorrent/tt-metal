// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile_hs_local.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

bool SliceTileHsLocalProgramFactory::applies(const SliceParams& args, const Tensor& input) {
    if (args.use_tensor_args || input.layout() != Layout::TILE || !input.is_sharded() ||
        input.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED ||
        !args.output_mem_config.is_sharded() ||
        args.output_mem_config.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED ||
        !args.output_mem_config.shard_spec().has_value()) {
        return false;
    }
    const auto& tile = input.tensor_spec().tile();
    if (tile.get_height() != TILE_HEIGHT || tile.get_width() != TILE_WIDTH) {
        return false;
    }
    for (uint32_t s : args.step) {
        if (s != 1) {
            return false;
        }
    }
    const auto& shape = input.logical_shape();
    const int rank = static_cast<int>(shape.rank());
    if (rank < 2 || static_cast<int>(args.slice_start.rank()) != rank || static_cast<int>(args.slice_end.rank()) != rank) {
        return false;
    }
    for (int d = 0; d < rank - 1; ++d) {  // only the last dimension may be cut
        if (args.slice_start[d] != 0 || args.slice_end[d] != shape[d]) {
            return false;
        }
    }
    const uint32_t w0 = args.slice_start[rank - 1];
    const uint32_t w1 = args.slice_end[rank - 1];
    if (w1 <= w0 || w1 > shape[rank - 1] || w0 % TILE_WIDTH != 0 || (w1 - w0) % TILE_WIDTH != 0) {
        return false;
    }
    const auto& iss = input.shard_spec().value();
    const auto& oss = args.output_mem_config.shard_spec().value();
    return iss.grid == oss.grid && iss.orientation == oss.orientation && iss.shape[0] == oss.shape[0] &&
           iss.shape[0] % TILE_HEIGHT == 0 && iss.shape[1] == input.padded_shape()[-1] && oss.shape[1] == w1 - w0;
}

tt::tt_metal::ProgramDescriptor SliceTileHsLocalProgramFactory::create_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    const auto& iss = input.shard_spec().value();
    const auto& oss = output.shard_spec().value();
    const int rank = static_cast<int>(input.logical_shape().rank());
    const uint32_t in_tile_bytes = static_cast<uint32_t>(input.buffer()->page_size());
    const uint32_t out_tile_bytes = static_cast<uint32_t>(output.buffer()->page_size());
    TT_FATAL(in_tile_bytes == out_tile_bytes, "slice (tile, local): input and output tile sizes differ ({} vs {})", in_tile_bytes, out_tile_bytes);
    const uint32_t ht = iss.shape[0] / TILE_HEIGHT;
    const uint32_t wt_in = iss.shape[1] / TILE_WIDTH;
    const uint32_t wt_out = oss.shape[1] / TILE_WIDTH;
    const uint32_t ct0 = args.slice_start[rank - 1] / TILE_WIDTH;

    ProgramDescriptor desc;
    constexpr uint8_t cb_in = tt::CBIndex::c_0;
    constexpr uint8_t cb_out = tt::CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = ht * wt_in * in_tile_bytes,
        .core_ranges = iss.grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_in, .data_format = datatype_to_dataformat_converter(input.dtype()), .page_size = in_tile_bytes}}},
        .buffer = input.buffer(),
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = ht * wt_out * out_tile_bytes,
        .core_ranges = oss.grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_out, .data_format = datatype_to_dataformat_converter(output.dtype()), .page_size = out_tile_bytes}}},
        .buffer = output.buffer(),
    });
    KernelDescriptor reader;
    reader.kernel_source = "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/slice_reader_tile_hs_local.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = iss.grid;
    reader.compile_time_args = {cb_in, cb_out, ht, wt_in, wt_out, ct0, in_tile_bytes};
    reader.config = ReaderConfigDescriptor{};
    desc.kernels.push_back(std::move(reader));
    return desc;
}

}  // namespace ttnn::prim
