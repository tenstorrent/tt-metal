// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_kv_cache_load_slice_program_factory.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

namespace {

// Returns per-core (start_id, writer_args) pairs.  The reader uses {src0_buffer, start_id};
// the buffer pointer is appended by the caller so it can be tracked as a BufferBinding for
// contract-1 cache-hit patching.
struct PerCoreSliceArgs {
    uint32_t reader_start_id;
    std::vector<uint32_t> writer_args;
};

std::vector<PerCoreSliceArgs> get_unpad_runtime_args_tile_sharded(
    const Tensor& input_tensor,
    const ttnn::Shape& output_tensor_start,
    uint32_t num_cores_total,
    uint32_t num_tiles_per_core) {
    auto input_shape = input_tensor.padded_shape();

    std::vector<PerCoreSliceArgs> ret_val(num_cores_total);

    uint32_t start_id = ttnn::operations::data_movement::get_tiled_start_offset(input_tensor, output_tensor_start);
    const uint32_t num_tiles_shifted_per_core = input_shape[-2] * input_shape[-1] / TILE_HW;

    for (uint32_t i = 0; i < num_cores_total; i++) {
        ret_val[i] = {start_id, std::vector<uint32_t>{num_tiles_per_core}};
        start_id += num_tiles_shifted_per_core;
    }

    return ret_val;
}

}  // namespace

tt::tt_metal::ProgramDescriptor NlpKVCacheLoadSliceProgramFactory::create_descriptor(
    const NlpKvCacheLoadSliceParams& operation_attributes,
    const NlpKvCacheLoadSliceInputs& tensor_args,
    Tensor& output) {
    ProgramDescriptor desc;

    const auto& a = tensor_args.input;
    const auto& output_tensor_start = operation_attributes.output_tensor_start;

    const auto& output_shape = output.padded_shape();
    const auto& input_shape = a.padded_shape();

    // This should allocate a DRAM buffer on the device
    auto shard_spec = output.shard_spec().value();
    auto all_cores = shard_spec.grid;
    auto num_cores_total = all_cores.num_cores();
    auto core_range = *all_cores.ranges().begin();
    auto num_cores_x = core_range.grid_size().x;
    uint32_t num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
    auto num_tiles_per_core = num_units_per_shard_height * num_units_per_shard_width;

    tt_metal::Buffer* src0_buffer = a.buffer();

    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t num_input_tiles = num_tiles_per_core;
    // Output-sharded CB: bind .buffer = output.buffer() so the framework re-applies
    // the dynamic CB address on every cache hit (replacing the explicit
    // UpdateDynamicCircularBufferAddress call from the legacy override path).
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = output.buffer(),
    });

    // Shared reader and writer config setup
    uint32_t num_unpadded_tiles_head_dim = output_shape[-1] / TILE_WIDTH;
    uint32_t num_unpadded_tiles_seqlen_dim = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_tiles_seqlen_dim =
        (input_shape[-2] / TILE_HEIGHT - num_unpadded_tiles_seqlen_dim) * (input_shape[-1] / TILE_WIDTH);

    // Reader compile-time args
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)num_tiles_per_core,
        (std::uint32_t)num_unpadded_tiles_head_dim,
        (std::uint32_t)num_unpadded_tiles_seqlen_dim,
        (std::uint32_t)num_padded_tiles_seqlen_dim,
        (std::uint32_t)num_cores_total};
    tt::tt_metal::TensorAccessorArgs(src0_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice/device/kernels/dataflow/"
        "reader_unary_unpad_dims_interleaved_start_id_shard_optimized.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer compile-time args
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)src0_cb_index};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    auto all_runtime_args =
        get_unpad_runtime_args_tile_sharded(a, output_tensor_start, num_cores_total, num_tiles_per_core);
    auto* input_buffer = a.buffer();

    reader_desc.runtime_args.reserve(num_cores_total);
    writer_desc.runtime_args.reserve(num_cores_total);
    for (uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i % num_cores_x, i / num_cores_x};
        reader_desc.emplace_runtime_args(core, {input_buffer, all_runtime_args[i].reader_start_id});
        writer_desc.runtime_args.emplace_back(core, std::move(all_runtime_args[i].writer_args));
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
