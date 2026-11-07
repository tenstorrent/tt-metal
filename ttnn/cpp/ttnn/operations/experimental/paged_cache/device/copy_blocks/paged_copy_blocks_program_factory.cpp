// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One data-movement kernel per core. The work is every (cache, src_dst row, page of a block) item, split into
// contiguous ranges, one range per core.

#include "paged_copy_blocks_program_factory.hpp"

#include <algorithm>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace cb {
constexpr uint32_t page = tt::CBIndex::c_0;   // one cache page in flight
constexpr uint32_t table = tt::CBIndex::c_1;  // the whole src_dst table
constexpr uint32_t zero = tt::CBIndex::c_2;   // one page of zeros
}  // namespace cb

ProgramDescriptor PagedCopyBlocksProgramFactory::create_descriptor(
    const PagedCopyBlocksParams& attrs, const PagedCopyBlocksInputs& in, std::vector<Tensor>& /*outputs*/) {
    const auto& caches = in.caches;
    Buffer* cache_buf = caches.front().buffer();
    Buffer* table_buf = in.src_dst.buffer();
    const uint32_t num_caches = caches.size();
    const uint32_t num_pairs = in.src_dst.logical_shape()[0];
    const uint32_t pages_per_block = attrs.pages_per_block;
    const uint32_t page_bytes = cache_buf->page_size();
    const uint32_t aligned_page_bytes = cache_buf->aligned_page_size();
    const uint32_t row_bytes = table_buf->aligned_page_size();

    auto* device = caches.front().device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t num_items = num_caches * num_pairs * pages_per_block;
    const uint32_t num_cores = std::min<uint32_t>(num_items, grid.x * grid.y);
    const uint32_t items_per_core = num_items / num_cores;
    const uint32_t cores_with_extra_item = num_items % num_cores;

    std::vector<CoreCoord> core_list(num_cores);
    std::set<CoreRange> core_set;
    for (uint32_t c = 0; c < num_cores; c++) {
        core_list[c] = CoreCoord{c / grid.y, c % grid.y};
        core_set.insert(CoreRange{core_list[c], core_list[c]});
    }
    CoreRangeSet cores{core_set};

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t idx, uint32_t total_bytes, uint32_t page_size, tt::DataFormat fmt) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = total_bytes,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = fmt, .page_size = page_size}}}});
    };
    const tt::DataFormat cache_fmt = datatype_to_dataformat_converter(caches.front().dtype());
    add_cb(cb::page, aligned_page_bytes, aligned_page_bytes, cache_fmt);
    add_cb(cb::zero, aligned_page_bytes, aligned_page_bytes, cache_fmt);
    add_cb(cb::table, num_pairs * row_bytes, row_bytes, datatype_to_dataformat_converter(in.src_dst.dtype()));

    std::vector<uint32_t> ct = {pages_per_block, num_pairs, page_bytes, aligned_page_bytes, row_bytes};
    TensorAccessorArgs(*cache_buf).append_to(ct);
    TensorAccessorArgs(*table_buf).append_to(ct);

    KernelDescriptor kernel;
    kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/paged_copy_blocks.cpp";
    kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = cores;
    kernel.compile_time_args = ct;
    kernel.config = ReaderConfigDescriptor{};
    kernel.runtime_args.reserve(num_cores);

    for (uint32_t c = 0, first_item = 0; c < num_cores; c++) {
        const uint32_t n_items = items_per_core + (c < cores_with_extra_item ? 1u : 0u);
        std::vector<std::variant<uint32_t, Buffer*>> args = {first_item, n_items, table_buf};
        for (const Tensor& cache : caches) {
            args.emplace_back(cache.buffer());
        }
        kernel.emplace_runtime_args(core_list[c], args);
        first_item += n_items;
    }

    desc.kernels.push_back(std::move(kernel));
    return desc;
}

}  // namespace ttnn::experimental::prim
