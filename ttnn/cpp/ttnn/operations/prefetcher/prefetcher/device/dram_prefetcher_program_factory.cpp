// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <tuple>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/global_circular_buffer.hpp>

#include "dram_prefetcher_program_factory.hpp"

namespace ttnn::prim {

using std::vector;

using namespace tt::tt_metal;

namespace {

std::pair<uint32_t, uint32_t> get_max_page_size_and_num_pages(
    uint32_t max_page_size, uint32_t num_tiles, uint32_t num_datums_per_tile) {
    uint64_t total_size = static_cast<uint64_t>(num_tiles) * num_datums_per_tile;

    uint32_t page_size = (max_page_size / num_datums_per_tile) * num_datums_per_tile;
    while (total_size % page_size != 0 && page_size >= num_datums_per_tile) {
        page_size -= num_datums_per_tile;
    }
    uint32_t num_pages = total_size / page_size;

    return {page_size, num_pages};
}

}  // namespace

DramPrefetcherGeometry compute_dram_prefetcher_geometry(
    const std::vector<Tensor>& weight_tensors, const uint32_t num_receivers_per_reader) {
    TT_FATAL(!weight_tensors.empty(), "dram_prefetcher needs at least one weight tensor besides the address tensor");
    DramPrefetcherGeometry geometry;
    geometry.num_receivers_per_reader = num_receivers_per_reader;
    geometry.num_readers = weight_tensors[0].shard_spec()->grid.num_cores();
    geometry.num_blocks = geometry.num_readers * num_receivers_per_reader;
    const uint32_t num_blocks = geometry.num_blocks;

    // Largest single NoC transfer the reader and writer issue.
    constexpr uint32_t max_page_size = 8192;

    geometry.tensors.reserve(weight_tensors.size());
    for (const auto& tensor : weight_tensors) {
        const tt::tt_metal::Tile tile = tensor.tensor_spec().tile();
        const auto& shard_shape = tensor.buffer()->shard_spec().shape();
        DramPrefetcherTensorGeometry t;
        t.data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
        t.tile_size = tile.get_tile_size(t.data_format);

        const uint32_t height_in_tiles = tt::round_up(shard_shape[0] / tile.get_tile_shape()[0], num_blocks);
        const uint32_t width_in_tiles = shard_shape[1] / tile.get_tile_shape()[1];
        t.block_height_in_tiles = height_in_tiles / num_blocks;
        t.block_num_tiles = height_in_tiles * width_in_tiles / num_blocks;

        std::tie(t.page_size, t.block_num_pages) =
            get_max_page_size_and_num_pages(max_page_size, t.block_num_tiles, tt::tile_size(t.data_format));
        std::tie(t.coalesced_page_size, t.coalesced_num_pages) = get_max_page_size_and_num_pages(
            max_page_size, width_in_tiles / num_receivers_per_reader, tt::tile_size(t.data_format));
        t.block_size_per_receiver = t.block_num_tiles * t.tile_size / num_receivers_per_reader;
        geometry.tensors.push_back(t);
    }

    for (const auto& t : geometry.tensors) {
        geometry.max_block_num_tiles = std::max(geometry.max_block_num_tiles, t.block_num_tiles);
        // The first tensor with the largest tile size sets the staging buffer's data format.
        if (t.tile_size > geometry.max_tile_size) {
            geometry.max_tile_size = t.tile_size;
            geometry.max_tile_size_data_format = t.data_format;
        }
    }
    geometry.max_block_size = geometry.max_tile_size * geometry.max_block_num_tiles;
    return geometry;
}

uint32_t dram_prefetcher_reader_vc(const std::vector<CoreCoord>& reader_cores, const uint32_t reader_index) {
    // Reader i reads DRAM bank i, on one of the two DRAM-read VCs (2 and 3) by bank parity. A reader
    // whose row already holds a reader of the same parity moves to the other VC.
    const auto parity_vc = [](uint32_t bank_id) { return (bank_id & 0x1) + 2; };
    const uint32_t bank_id = reader_index;
    uint32_t vc = parity_vc(bank_id);
    for (uint32_t j = 0; j < reader_index; ++j) {
        if (reader_cores[j].y == reader_cores[reader_index].y && parity_vc(bank_id) == parity_vc(j)) {
            vc = ((vc + 1) & 0x1) + 2;
            break;
        }
    }
    return vc;
}

ProgramDescriptor DramPrefetcherProgramFactory::create_descriptor(
    const DramPrefetcherParams& operation_attributes,
    const DramPrefetcherInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    const auto& input_tensors = tensor_args.input_tensors;
    TT_FATAL(!input_tensors.empty(), "Must have at least one input tensor");
    TT_FATAL(operation_attributes.global_cb.has_value(), "Global circular buffer must be provided");
    const auto& global_cb = *(operation_attributes.global_cb);
    const uint32_t num_layers = operation_attributes.num_layers;
    const bool enable_performance_mode = operation_attributes.enable_performance_mode;

    /* Buffers */
    const Buffer& global_cb_buffer = global_cb.cb_buffer();
    // tensors that with addresses
    const ttnn::Tensor& tensor_addrs = input_tensors.back();  // Last tensor is tensor_addrs
    Buffer* tensor_addrs_buffer = tensor_addrs.buffer();
    // tensors that with actual data
    const std::vector<Tensor> tensors(input_tensors.begin(), input_tensors.end() - 1);

    /* Dataformats */
    tt::DataFormat tensor_addrs_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_addrs.dtype());

    // In validate we make sure that all tensors are on the same device
    uint32_t num_tensors = tensors.size();
    auto sender_receiver_core_mapping = global_cb.sender_receiver_core_mapping()[0];
    uint32_t num_receivers_per_reader = sender_receiver_core_mapping.second.num_cores();

    const DramPrefetcherGeometry geometry = compute_dram_prefetcher_geometry(tensors, num_receivers_per_reader);
    const uint32_t num_readers = geometry.num_readers;
    const uint32_t num_blocks = geometry.num_blocks;
    const uint32_t max_block_tiles = geometry.max_block_num_tiles;
    const uint32_t max_tile_size = geometry.max_tile_size;
    const tt::DataFormat max_tile_size_df = geometry.max_tile_size_data_format;
    const uint32_t max_block_size_per_reader_core = geometry.max_block_size;

    uint32_t max_tensor_size = max_block_size_per_reader_core / num_receivers_per_reader * num_blocks;

    TT_FATAL(
        max_tensor_size <= global_cb.size(),
        "largest tensor {} must fit in global cb {}",
        max_tensor_size,
        global_cb.size());

    /* Cores setup */
    const auto& all_reader_core_range = global_cb.sender_cores();
    auto reader_core_range_vec = corerange_to_cores(all_reader_core_range, std::nullopt, true);
    std::vector<CoreRange> active_reader_core_range_vec;
    active_reader_core_range_vec.reserve(num_readers);
    for (uint32_t i = 0; i < num_readers; ++i) {
        auto core = reader_core_range_vec[i];
        active_reader_core_range_vec.push_back(CoreRange{core, core});
    }
    auto reader_core_range = CoreRangeSet{std::move(active_reader_core_range_vec)};

    /* read cb setup */
    uint32_t reader_cb_single_tile_size = max_tile_size;
    const uint32_t total_num_blocks_in_buffer = kDramPrefetcherStagingBlocks;
    uint32_t reader_cb_size = max_block_size_per_reader_core * total_num_blocks_in_buffer;

    TT_FATAL(reader_cb_size <= global_cb.size(), "reader_cb_size must not be larger than global cb");

    uint32_t reader_cb_index = tt::CBIndex::c_0;
    uint32_t sync_cb_index = tt::CBIndex::c_3;
    uint32_t sync_cb_page_size = hal::get_l1_alignment();

    /* tensor addresses cb setup */
    uint32_t tensor_addrs_single_tile_size = sizeof(uint32_t);
    uint32_t tensor_addrs_cb_size = num_layers * num_tensors * tensor_addrs_single_tile_size;

    uint32_t tensor_addrs_cb_index = tt::CBIndex::c_1;

    /* remote cb setup */
    uint32_t remote_cb_size = global_cb.size();

    auto L1_ALIGNMENT = tt::tt_metal::hal::get_l1_alignment();
    uint32_t remote_cb_index = tt::CBIndex::c_31;

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = reader_cb_size,
        .core_ranges = reader_core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reader_cb_index),
            .data_format = max_tile_size_df,
            .page_size = reader_cb_single_tile_size,
        }}},
        .buffer = const_cast<Buffer*>(std::addressof(global_cb_buffer)),
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = sync_cb_page_size,
        .core_ranges = reader_core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(sync_cb_index),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = sync_cb_page_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = tensor_addrs_cb_size,
        .core_ranges = reader_core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tensor_addrs_cb_index),
            .data_format = tensor_addrs_data_format,
            .page_size = tensor_addrs_single_tile_size,
        }}},
        .buffer = tensor_addrs_buffer,
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = remote_cb_size,
        .core_ranges = reader_core_range,
        .remote_format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(remote_cb_index),
            .data_format = max_tile_size_df,
            .page_size = L1_ALIGNMENT,  // set to 16B so that the infra won't update write pointers to wrong location
        }}},
        .global_circular_buffer = std::addressof(global_cb),
    });

    // Reader kernel
    std::vector<uint32_t> reader_ct_args = {
        num_layers,
        num_tensors,
        num_blocks,
        reader_cb_size,
        max_block_tiles,
        max_block_size_per_reader_core,
        reader_cb_index,
        tensor_addrs_cb_index,
        sync_cb_index,
    };
    reader_ct_args.push_back(static_cast<uint32_t>(enable_performance_mode));

    // Writer kernel
    std::vector<uint32_t> writer_ct_args = {
        num_layers,
        num_tensors,
        num_blocks,
        num_receivers_per_reader,
        max_block_tiles,
        reader_cb_index,
        remote_cb_index,
        sync_cb_index,
    };
    writer_ct_args.push_back(static_cast<uint32_t>(enable_performance_mode));

    /* Runtime args */
    std::vector<uint32_t> page_sizes;
    std::vector<uint32_t> block_num_pages;
    std::vector<uint32_t> tensor_block_num_tiles;
    std::vector<uint32_t> coalesced_page_sizes;
    std::vector<uint32_t> coalesced_num_pages;
    std::vector<uint32_t> tensor_tile_sizes;
    std::vector<uint32_t> block_heights_in_tiles;
    for (const auto& t : geometry.tensors) {
        page_sizes.push_back(t.page_size);
        block_num_pages.push_back(t.block_num_pages);
        tensor_block_num_tiles.push_back(t.block_num_tiles);
        coalesced_page_sizes.push_back(t.coalesced_page_size);
        coalesced_num_pages.push_back(t.coalesced_num_pages);
        tensor_tile_sizes.push_back(t.tile_size);
        block_heights_in_tiles.push_back(t.block_height_in_tiles);
    }

    const auto& reader_cores = corerange_to_cores(reader_core_range, std::nullopt, true);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/reader_dram.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = reader_core_range;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_0_default,
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
    };

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/writer_l1.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = reader_core_range;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
    };

    // Runtime args for the reader cores
    for (uint32_t core_index = 0; core_index < reader_core_range.num_cores(); ++core_index) {
        const auto& core = reader_cores[core_index];

        /* reader kernel */
        const uint32_t bank_id = core_index;
        const uint32_t vc = dram_prefetcher_reader_vc(reader_cores, core_index);

        std::vector<uint32_t> reader_rt_args;
        reader_rt_args.reserve(3 + 3 * num_tensors);
        reader_rt_args.insert(reader_rt_args.end(), {bank_id, vc, total_num_blocks_in_buffer});
        reader_rt_args.insert(reader_rt_args.end(), page_sizes.begin(), page_sizes.end());
        reader_rt_args.insert(reader_rt_args.end(), block_num_pages.begin(), block_num_pages.end());
        reader_rt_args.insert(reader_rt_args.end(), tensor_block_num_tiles.begin(), tensor_block_num_tiles.end());

        reader_desc.runtime_args.emplace_back(core, std::move(reader_rt_args));

        /* writer kernel */
        std::vector<uint32_t> writer_rt_args;
        writer_rt_args.reserve(5 * num_tensors);
        writer_rt_args.insert(writer_rt_args.end(), coalesced_page_sizes.begin(), coalesced_page_sizes.end());
        writer_rt_args.insert(writer_rt_args.end(), coalesced_num_pages.begin(), coalesced_num_pages.end());
        writer_rt_args.insert(writer_rt_args.end(), tensor_block_num_tiles.begin(), tensor_block_num_tiles.end());
        writer_rt_args.insert(writer_rt_args.end(), tensor_tile_sizes.begin(), tensor_tile_sizes.end());
        writer_rt_args.insert(writer_rt_args.end(), block_heights_in_tiles.begin(), block_heights_in_tiles.end());

        writer_desc.runtime_args.emplace_back(core, std::move(writer_rt_args));
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
