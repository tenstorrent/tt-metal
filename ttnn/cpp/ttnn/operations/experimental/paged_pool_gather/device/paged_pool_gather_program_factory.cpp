// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Program factory of paged_pool_gather: work items are (user, compact output block) pairs dealt
// round-robin over the cores; one data-movement kernel per core reads the pools of its block into an
// L1 tile block (double-buffered) and writes the block's tiles to the output.

#include "paged_pool_gather_program_factory.hpp"

#include <algorithm>
#include <set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::experimental::paged_pool_gather {

tt::tt_metal::ProgramDescriptor PagedPoolGatherProgramFactory::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& in, Tensor& output) {
    const auto& cs = in.cache.logical_shape();
    const uint32_t Wt = cs[3] / TILE_WIDTH;
    const uint32_t B = in.pool_ids.logical_shape()[0];
    const uint32_t blocks_per_seq = in.page_table.logical_shape()[1];
    const uint32_t P = attrs.pool_size;
    const uint32_t N = attrs.num_pools;
    const uint32_t out_blocks = attrs.out_blocks;
    const uint32_t items = B * out_blocks;

    auto* device = in.cache.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t n_cores_grid = grid.x * grid.y;
    const uint32_t NC = std::max<uint32_t>(1u, std::min<uint32_t>(items, n_cores_grid));
    std::vector<CoreCoord> core_of(NC);
    std::set<CoreRange> core_set;
    for (uint32_t c = 0; c < NC; c++) {
        core_of[c] = CoreCoord{c / grid.y, c % grid.y};
        core_set.insert(CoreRange{core_of[c], core_of[c]});
    }
    CoreRangeSet cores{core_set};

    const tt::DataFormat f16 = tt::DataFormat::Float16_b;
    const uint32_t tile_bytes = tt::tile_size(f16);  // 2048
    const uint32_t pt_row_bytes = in.page_table.buffer()->aligned_page_size();
    const uint32_t ids_row_bytes = in.pool_ids.buffer()->aligned_page_size();
    const uint32_t cache_page_bytes = in.cache.buffer()->aligned_page_size();
    const uint32_t out_page_bytes = output.buffer()->aligned_page_size();
    TT_FATAL(
        cache_page_bytes == tile_bytes && out_page_bytes == tile_bytes, "paged_pool_gather: bf16 tile pages expected");

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t idx, uint32_t page, uint32_t n_pages) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = n_pages * page,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = f16, .page_size = page}}}});
    };
    // cb 0: two tile blocks + one spare tile for 64 B alignment slack; cb 1: the user's pool-id row;
    // cb 2: one 64 B page-table fragment per pool slot of a block (x2 for alignment slack)
    add_cb(0, tile_bytes, 2 * Wt + 1);
    const uint32_t ids_cb_bytes = ((ids_row_bytes + 63) / 64) * 64 + 64;
    add_cb(1, 64, ids_cb_bytes / 64);
    const uint32_t slots_per_block = TILE_HEIGHT / P;
    add_cb(2, 64, 2 * slots_per_block + 1);

    std::vector<uint32_t> ct = {Wt, P, N, out_blocks, blocks_per_seq, pt_row_bytes, ids_row_bytes, slots_per_block};
    TensorAccessorArgs(*in.cache.buffer()).append_to(ct);
    TensorAccessorArgs(*in.page_table.buffer()).append_to(ct);
    TensorAccessorArgs(*in.pool_ids.buffer()).append_to(ct);
    TensorAccessorArgs(*output.buffer()).append_to(ct);

    KernelDescriptor k;
    k.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/paged_pool_gather/device/kernels/dataflow/paged_pool_gather.cpp";
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.compile_time_args = ct;
    k.config = ReaderConfigDescriptor{};
    k.runtime_args.reserve(NC);
    auto* cache_buf = in.cache.buffer();
    auto* pt_buf = in.page_table.buffer();
    auto* ids_buf = in.pool_ids.buffer();
    auto* out_buf = output.buffer();
    for (uint32_t c = 0; c < NC; c++) {
        const uint32_t n_items = (items - c + NC - 1) / NC;
        k.emplace_runtime_args(core_of[c], {c, n_items, NC, cache_buf, pt_buf, ids_buf, out_buf});
    }
    desc.kernels.push_back(std::move(k));
    return desc;
}

}  // namespace ttnn::operations::experimental::paged_pool_gather
