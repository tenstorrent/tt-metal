// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_sharded_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <algorithm>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor TransposeWHShardedProgramFactory::create_descriptor(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    ProgramDescriptor desc;

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const auto tile = input_tensor.tensor_spec().tile();
    const uint32_t tile_hw = tile.get_tile_hw();

    IDevice* device = input_tensor.device();

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto shard_spec = input_tensor.shard_spec().value();
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    auto& all_cores = shard_spec.grid;
    uint32_t num_tiles_per_shard = shard_spec.numel() / tile_hw;

    // Sharded CBs: total_size depends on num_tiles_per_shard which can vary across
    // cache hits; .buffer triggers UpdateDynamicCircularBufferAddress. total_size
    // is not in the program hash so the framework re-applies the combined update
    // via apply_descriptor_runtime_args.
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = num_tiles_per_shard;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src0_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }}},
        .buffer = input_tensor.buffer(),
    });

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = num_tiles_per_shard;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = total_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
        .buffer = output_tensor.buffer(),
    });

    std::vector<uint32_t> reader_compile_time_args = {src0_cb_index};
    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    std::vector<uint32_t> compute_compile_time_args = {src0_cb_index, output_cb_index};

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = total_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = total_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (src0_cb_data_format == tt::DataFormat::Float32) {
        unpack_to_dest_mode[src0_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_sharded.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = total_cores;
    compute_desc.compile_time_args = std::move(compute_compile_time_args);
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    auto padded_shape = input_tensor.padded_shape();
    auto shard_shape = shard_spec.shape;

    uint32_t H = padded_shape[2];
    uint32_t Hs = shard_shape[0], Ws = shard_shape[1];

    uint32_t Hts = Hs / tile.get_height();
    uint32_t Wts = Ws / tile.get_width();

    uint32_t Ht = H / tile.get_height();
    uint32_t Ht_per_shard = std::min(Ht, Hts);

    uint32_t num_hw_blocks_per_shard = Hts > Ht ? Hts / Ht : 1;

    uint32_t HtWt_tile_size = Ht_per_shard * Wts;
    uint32_t num_blocks = num_hw_blocks_per_shard * HtWt_tile_size;

    auto bbox = all_cores.bounding_box();
    std::vector<CoreCoord> cores =
        grid_to_cores_with_noop(bbox.end_coord.x, bbox.end_coord.y, num_cores_x, num_cores_y, row_major);

    // Active shard cores get the real arg values; the trailing no-op cores keep the
    // default-constructed slots (matching legacy std::fill behavior on cores.size()).
    const std::vector<uint32_t> reader_rt = {num_blocks};
    const std::vector<uint32_t> compute_rt = {num_blocks, HtWt_tile_size, num_hw_blocks_per_shard, Ht_per_shard, Wts};
    const std::vector<uint32_t> writer_rt = {num_blocks};

    const uint32_t num_active = all_cores.num_cores();
    reader_desc.runtime_args.reserve(cores.size());
    compute_desc.runtime_args.reserve(cores.size());
    writer_desc.runtime_args.reserve(cores.size());
    for (uint32_t i = 0; i < cores.size(); ++i) {
        if (i < num_active) {
            reader_desc.runtime_args.emplace_back(cores[i], reader_rt);
            compute_desc.runtime_args.emplace_back(cores[i], compute_rt);
            writer_desc.runtime_args.emplace_back(cores[i], writer_rt);
        } else {
            // No-op core: matches legacy std::vector<uint32_t>(1)/(5) zero-initialized rows.
            reader_desc.runtime_args.emplace_back(cores[i], std::vector<uint32_t>(1));
            compute_desc.runtime_args.emplace_back(cores[i], std::vector<uint32_t>(5));
            writer_desc.runtime_args.emplace_back(cores[i], std::vector<uint32_t>(1));
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
