// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/span.hpp>

#include "dram_prefetcher_device_operation_types.hpp"

namespace ttnn::prim {

// One weight tensor's share of a dram_prefetcher run, as every reader core moves it: the shard is cut
// into `num_blocks` blocks of `block_height_in_tiles` tile rows, each read from DRAM in
// `block_num_pages` pages of `page_size` bytes, and each split column-wise across the reader's receivers
// in rows of `coalesced_num_pages` writes of `coalesced_page_size` bytes.
struct DramPrefetcherTensorGeometry {
    tt::DataFormat data_format = tt::DataFormat::Invalid;
    uint32_t tile_size = 0;
    uint32_t block_height_in_tiles = 0;
    uint32_t block_num_tiles = 0;
    // DRAM read pages of one block.
    uint32_t page_size = 0;
    uint32_t block_num_pages = 0;
    // One receiver's slice of one block row.
    uint32_t coalesced_page_size = 0;
    uint32_t coalesced_num_pages = 0;
    // One receiver's slice of one block: what each receiver is delivered per block.
    uint32_t block_size_per_receiver = 0;
};

// The block geometry dram_prefetcher derives from its weight tensors and its sender -> receivers layout.
// Shared by the GlobalCircularBuffer and the PrefetcherPipe program factories.
struct DramPrefetcherGeometry {
    // Blocks each reader pushes per tensor: one per receiver of the whole ring.
    uint32_t num_blocks = 0;
    std::vector<DramPrefetcherTensorGeometry> tensors;
    uint32_t max_block_num_tiles = 0;
    uint32_t max_tile_size = 0;
    tt::DataFormat max_tile_size_data_format = tt::DataFormat::Invalid;
    // A reader's staging slot: the largest block of any tensor at the largest tile size.
    uint32_t max_block_size = 0;
};

// `weight_tensors` are dram_prefetcher's inputs without the trailing address tensor.
DramPrefetcherGeometry compute_dram_prefetcher_geometry(
    ttsl::Span<const Tensor> weight_tensors, uint32_t num_receivers_per_reader);

// Appends `field` of every tensor in `geometry` to `args`: one per-tensor table of the kernels' runtime
// args, which index it as table * num_tensors + tensor.
void append_per_tensor_args(
    std::vector<uint32_t>& args,
    const DramPrefetcherGeometry& geometry,
    uint32_t DramPrefetcherTensorGeometry::* field);

// The cores dram_prefetcher reads on, taken from its delivery target.
struct DramPrefetcherReaders {
    // In reader order: reader i reads DRAM bank i.
    std::vector<tt::tt_metal::CoreCoord> cores;
    // Every reader's receiver count.
    uint32_t num_receivers_per_reader = 0;
};

// The first `num_readers` senders of whichever target `params` delivers into: `global_cb`'s, in
// sender_receiver_core_mapping order, or `prefetcher_pipes`', in the order given. TT_FATALs unless there are
// that many, each on its own core and all with the same number of receivers. `num_readers` must be nonzero.
DramPrefetcherReaders dram_prefetcher_readers(const DramPrefetcherParams& params, uint32_t num_readers);

// The NoC virtual channel reader `reader_index` of `reader_cores` reads its DRAM bank on, chosen so that
// readers on one row alternate channels.
uint32_t dram_prefetcher_reader_vc(const std::vector<tt::tt_metal::CoreCoord>& reader_cores, uint32_t reader_index);

// Reader blocks the staging buffer holds: one being filled, one in flight, one draining.
constexpr uint32_t kDramPrefetcherStagingBlocks = 3;

// Delivers into a worker-sender GlobalCircularBuffer (`global_cb`).
struct DramPrefetcherProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const DramPrefetcherParams& operation_attributes,
        const DramPrefetcherInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
