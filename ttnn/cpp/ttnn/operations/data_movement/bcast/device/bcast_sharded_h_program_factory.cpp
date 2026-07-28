// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_sharded_h_program_factory.hpp"

#include <cmath>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"
#include <tt-metalium/tilize_utils.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::constants;

tt::tt_metal::ProgramDescriptor BcastShardedHProgramFactory::create_descriptor(
    const BcastParams& operation_attributes, const BcastInputs& tensor_args, Tensor& tensor_return_value) {
    const Tensor& a = tensor_args.input_a;
    const Tensor& b = tensor_args.input_b;
    Tensor& output = tensor_return_value;

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    const uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    const uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    const uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    const uint32_t NC = N * C;

    const auto& tile = a.tensor_spec().tile();
    const uint32_t tile_h = tile.get_height();
    const uint32_t tile_w = tile.get_width();

    IDevice* device = a.device();

    const auto shard_spec = a.shard_spec().value();
    const auto all_cores = shard_spec.grid;
    const uint32_t ncores = shard_spec.num_cores();

    uint32_t ncores_x = device->compute_with_storage_grid_size().x;

    const auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(
        out_shard_spec.num_cores() == ncores,
        "Output tensor should have same number of cores {} as input tensor {}",
        out_shard_spec.num_cores(),
        ncores);

    const auto act_df = datatype_to_dataformat_converter(a.dtype());
    const auto b_df = datatype_to_dataformat_converter(b.dtype());
    const auto out_df = datatype_to_dataformat_converter(output.dtype());

    const uint32_t input_tile_size = tile.get_tile_size(act_df);
    const uint32_t input1_tile_size = tile.get_tile_size(b_df);
    const uint32_t output_tile_size = tile.get_tile_size(out_df);

    TT_FATAL(input_tile_size == output_tile_size, "Input and output tile size should be same");

    const uint32_t ntiles_along_width = std::ceil(shard_spec.shape[1] / (float)tile_w);
    const uint32_t ntiles_along_height = std::ceil(shard_spec.shape[0] / (float)tile_h);
    const uint32_t num_tile_per_core = ntiles_along_width * ntiles_along_height;

    uint32_t Wt, Ht;
    if (a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        ncores_x = all_cores.ranges().begin()->end_coord.y + 1;
        Wt = shard_spec.shape[1] / tile_w;
        Ht = shard_spec.shape[0] / tile_h;
    } else if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        Wt = shard_spec.shape[1] / tile_w;
        Ht = shard_spec.shape[0] / tile_h;
        TT_ASSERT(
            (shard_spec.shape[0] % (bN * tile_h) == 0),
            "Shard height per batch must be divisible by tile height {} {} {} ",
            shard_spec.shape[0],
            bN,
            tile_h);
    } else {
        TT_THROW("Unsupported memory layout");
    }

    TT_ASSERT(
        (shard_spec.shape[0] % tile_h == 0) && (shard_spec.shape[1] % tile_w == 0),
        "Shard shapes must be multiple of tile dimensions");

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    const uint32_t aligned_input_tile_nbytes =
        round_up_to_mul32(input_tile_size);  // will have issue if the page is not multiple of 32
    const uint32_t in_cb_pagesize = aligned_input_tile_nbytes;

    const uint32_t output_cb_index = tt::CBIndex::c_16;

    const uint32_t num_input_tiles =
        (b.padded_shape()[-1] * output.element_size() + tile.get_tile_hw() - 1) / tile.get_tile_hw();
    const uint32_t src1_cb_index = tt::CBIndex::c_1;

    Buffer* src0_buffer = a.buffer();
    Buffer* src1_buffer = b.buffer();
    Buffer* dst_buffer = output.buffer();

    ProgramDescriptor desc;

    // ---- Circular buffers ----

    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_input_tile_nbytes * num_tile_per_core,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = act_df,
            .page_size = in_cb_pagesize,
            .tile = TileDescriptor(tile),
        }}},
        .buffer = src0_buffer,
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * input1_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = b_df,
            .page_size = input1_tile_size,
            .tile = TileDescriptor(tile),
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_input_tile_nbytes * num_tile_per_core,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = out_df,
            .page_size = in_cb_pagesize,
            .tile = TileDescriptor(tile),
        }}},
        .buffer = dst_buffer,
    });

    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_cb_index};
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    const bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};
    (void)dst_is_dram;
    (void)writer_compile_time_args;

    const std::map<std::string, std::string> bcast_defines =
        bcast_op_utils::get_defines(BcastOpDim::H, operation_attributes.math_op);

    static constexpr const char* READER_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/reader_bcast_h_sharded.cpp";
    static constexpr const char* BCAST_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_h.cpp";

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = BCAST_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.defines = {bcast_defines.begin(), bcast_defines.end()};
    compute_desc.config = ComputeConfigDescriptor{};

    const uint32_t ncores_y = ncores / ncores_x;
    for (uint32_t i = 0; i < ncores; i++) {
        CoreCoord core;
        uint32_t offset = 0;
        uint32_t Ht_per_core = 0;
        if (a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            core = {i / ncores_x, i % ncores_x};
            Ht_per_core = Wt * Ht;
            if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
                offset = Wt * (i / ncores_x) + Wt * ncores_y * ((i % ncores_x) / (ncores_x / bN));
            } else {
                offset = Wt * (i % ncores_x) + Wt * ncores_x * ((i / ncores_x) / (ncores_y / bN));
            }
        } else if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            core = {i % ncores_x, i / ncores_x};
            if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
                offset = Wt * (core.x + core.y * ncores_x);
            } else {
                offset = Wt * (ncores_y * core.x + core.y);
                if (core.y == ncores_y) {
                    offset = Wt * (ncores_y * ncores_x + core.x);
                }
            }
            Ht_per_core = Ht / bN;
        }
        const uint32_t tile_offset = Wt * ncores;  // used in multi batch weight for block sharded
        reader_desc.emplace_runtime_args(
            core,
            {
                b.buffer(),   // 0
                Ht,           // 1
                Wt,           // 2
                offset,       // 3
                Ht_per_core,  // 4
                tile_offset,  // 5
            });

        compute_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                NC,  // B
                Ht,  // Hbatch  for block sharded
                Wt,  // Wt
            });
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
