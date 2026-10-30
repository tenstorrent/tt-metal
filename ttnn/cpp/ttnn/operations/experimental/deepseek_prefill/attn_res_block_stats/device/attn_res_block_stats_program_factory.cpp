// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/device/attn_res_block_stats_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

constexpr auto kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/device/kernels/";

CBDescriptor make_cb(
    CBIndex index, const CoreRangeSet& cores, uint32_t num_tiles, uint32_t tile_size, DataFormat format) {
    return CBDescriptor{
        .total_size = num_tiles * tile_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = format,
            .page_size = tile_size,
        }}},
    };
}

}  // namespace

tt::tt_metal::ProgramDescriptor AttnResBlockStatsProgramFactory::create_descriptor(
    const AttnResBlockStatsParams& operation_attributes,
    const AttnResBlockStatsInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* device = tensor_args.queries.device();
    const auto& first = tensor_args.sealed.front();

    const auto in_format = datatype_to_dataformat_converter(first.dtype());
    const auto in_tile_size = tt::tile_size(in_format);
    const auto out_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    const auto out_tile_size = tt::tile_size(out_format);

    const auto& shape = first.padded_shape();
    const uint32_t Ht = shape[2] / TILE_HEIGHT;
    const uint32_t Wt = shape[3] / TILE_WIDTH;
    const uint32_t num_candidates = tensor_return_value.padded_shape()[1];

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    // A work item is one candidate's token row-tile: its whole row of d, which is what
    // both reductions run over, so an item's output tile is complete when it is done.
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_items = num_candidates * Ht;
    const uint32_t num_cores = std::min<uint32_t>(num_items, grid.x * grid.y);
    const CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);

    ProgramDescriptor desc;

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    // A row is one CB unit, double-buffered so the next item's row arrives while this one
    // is reduced. The squares hold a row too: the sum of squares is a second matmul over
    // them, into the same accumulator as the dots.
    desc.cbs.push_back(make_cb(CBIndex::c_0, all_cores, 2 * Wt, in_tile_size, in_format));  // v
    desc.cbs.push_back(make_cb(CBIndex::c_1, all_cores, Wt, in_tile_size, in_format));      // queries, resident
    desc.cbs.push_back(make_cb(CBIndex::c_2, all_cores, 1, in_tile_size, in_format));       // ones in column 0
    desc.cbs.push_back(make_cb(CBIndex::c_3, all_cores, Wt, in_tile_size, in_format));      // v squared
    desc.cbs.push_back(make_cb(CBIndex::c_16, all_cores, 2, out_tile_size, out_format));    // statistics

    ////////////////////////////////////////////////////////////////////////////
    //                         Kernels
    ////////////////////////////////////////////////////////////////////////////
    // A core finds its items from its own position, so a launch ships no per-core args.
    const std::vector<uint32_t> split_ct_args = {grid.x, num_cores, num_items};

    std::vector<uint32_t> reader_ct_args = {Ht, Wt};
    reader_ct_args.insert(reader_ct_args.end(), split_ct_args.begin(), split_ct_args.end());
    TensorAccessorArgs(*first.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*tensor_args.queries.buffer()).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {};
    writer_ct_args.insert(writer_ct_args.end(), split_ct_args.begin(), split_ct_args.end());
    TensorAccessorArgs(*tensor_return_value.buffer()).append_to(writer_ct_args);

    std::vector<uint32_t> compute_ct_args = {Wt};
    compute_ct_args.insert(compute_ct_args.end(), split_ct_args.begin(), split_ct_args.end());

    // Each candidate is a (snapshot address, first page) pair: the snapshot that holds it and
    // the offset of its plane there. The snapshots share one accessor configuration, which
    // the device operation checks, so the address is all that differs between them.
    KernelDescriptor::RTArgList reader_common;
    reader_common.push_back(tensor_args.queries.buffer());
    for (const auto& snapshot : tensor_args.sealed) {
        for (uint32_t plane = 0; plane < snapshot.padded_shape()[1]; ++plane) {
            reader_common.push_back(snapshot.buffer());
            reader_common.push_back(plane * Ht * Wt);
        }
    }

    KernelDescriptor reader;
    reader.kernel_source = std::string(kKernelDir) + "reader_attn_res_block_stats.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct_args);
    reader.config = ReaderConfigDescriptor{};
    reader.emplace_common_runtime_args(reader_common);

    KernelDescriptor writer;
    writer.kernel_source = std::string(kKernelDir) + "writer_attn_res_block_stats.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct_args);
    writer.config = WriterConfigDescriptor{};
    writer.emplace_common_runtime_args({tensor_return_value.buffer()});

    KernelDescriptor compute;
    compute.kernel_source = std::string(kKernelDir) + "attn_res_block_stats.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = std::move(compute_ct_args);
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));

    return desc;
}

}  // namespace ttnn::experimental::prim
