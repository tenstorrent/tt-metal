// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/global_circular_buffer.hpp>

#include "dram_prefetcher_device_operation.hpp"

namespace ttnn::prim {

using std::vector;

using namespace tt::tt_metal;

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

ProgramDescriptor DramPrefetcherOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
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
    std::vector<Buffer*> tensor_buffers;
    // tensors that with actual data
    std::vector<Tensor> tensors;
    tensors.resize(input_tensors.size() - 1);
    std::copy(input_tensors.begin(), input_tensors.end() - 1, tensors.begin());
    tensor_buffers.reserve(tensors.size());
    std::transform(
        tensors.begin(), tensors.end(), std::back_inserter(tensor_buffers), [](const auto& t) { return t.buffer(); });

    /* Tiles */
    std::vector<tt::tt_metal::Tile> tensor_tiles;
    tensor_tiles.reserve(tensors.size());
    std::transform(tensors.begin(), tensors.end(), std::back_inserter(tensor_tiles), [](const auto& t) {
        return t.tensor_spec().tile();
    });

    /* Dataformats */
    tt::DataFormat tensor_addrs_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_addrs.dtype());
    std::vector<tt::DataFormat> tensor_data_formats;
    tensor_data_formats.reserve(tensors.size());
    std::transform(tensors.begin(), tensors.end(), std::back_inserter(tensor_data_formats), [](const auto& t) {
        return tt::tt_metal::datatype_to_dataformat_converter(t.dtype());
    });

    // In validate we make sure that all tensors are on the same device
    uint32_t num_tensors = tensors.size();
    auto sender_receiver_core_mapping = global_cb.sender_receiver_core_mapping()[0];
    uint32_t num_receivers_per_reader = sender_receiver_core_mapping.second.num_cores();

    uint32_t num_readers = tensors[0].shard_spec()->grid.num_cores();
    uint32_t num_blocks = num_readers * num_receivers_per_reader;

    std::vector<uint32_t> tensor_block_num_tiles;
    std::vector<std::vector<uint32_t>> tensor_shapes(num_tensors, std::vector<uint32_t>(2));
    std::vector<uint32_t> tensor_tile_sizes;
    tensor_block_num_tiles.reserve(num_tensors);
    tensor_tile_sizes.reserve(num_tensors);
    for (uint32_t t = 0; t < num_tensors; t++) {
        uint32_t height_in_tiles = tensor_buffers[t]->shard_spec().shape()[0] / tensor_tiles[t].get_tile_shape()[0];
        uint32_t width_in_tiles = tensor_buffers[t]->shard_spec().shape()[1] / tensor_tiles[t].get_tile_shape()[1];

        height_in_tiles = tt::round_up(height_in_tiles, num_blocks);
        tensor_shapes[t][0] = height_in_tiles;
        tensor_shapes[t][1] = width_in_tiles;
        tensor_block_num_tiles.push_back(height_in_tiles * width_in_tiles / num_blocks);
        tensor_tile_sizes.push_back(tensor_tiles[t].get_tile_size(tensor_data_formats[t]));
    }
    uint32_t max_block_tiles = *std::max_element(tensor_block_num_tiles.begin(), tensor_block_num_tiles.end());
    auto max_tile_size_iterator = std::max_element(tensor_tile_sizes.begin(), tensor_tile_sizes.end());
    uint32_t max_tile_size = *max_tile_size_iterator;
    uint32_t max_tile_size_tensor_idx = std::distance(tensor_tile_sizes.begin(), max_tile_size_iterator);
    tt::DataFormat max_tile_size_df = tensor_data_formats[max_tile_size_tensor_idx];

    uint32_t max_block_size_per_reader_core = max_tile_size * max_block_tiles;
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
    for (uint32_t i = 0; i < num_readers; ++i) {
        auto core = reader_core_range_vec[i];
        active_reader_core_range_vec.push_back(CoreRange{core, core});
    }
    auto reader_core_range = CoreRangeSet{active_reader_core_range_vec};

    /* read cb setup */
    uint32_t reader_cb_single_tile_size = max_tile_size;
    const uint32_t total_num_blocks_in_buffer = 3;  // reader cb is triple buffered
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

    std::vector<uint32_t> coalesced_page_sizes;
    std::vector<uint32_t> coalesced_num_pages;

    uint32_t max_page_size = 8192;

    for (uint32_t t = 0; t < num_tensors; t++) {
        auto [page_size, num_pages] = get_max_page_size_and_num_pages(
            max_page_size, tensor_block_num_tiles[t], tt::tile_size(tensor_data_formats[t]));
        page_sizes.push_back(page_size);
        block_num_pages.push_back(num_pages);

        uint32_t block_width_in_tiles = tensor_shapes[t][1];
        auto [coalesced_page_size, coalesced_num_page] = get_max_page_size_and_num_pages(
            max_page_size, block_width_in_tiles / num_receivers_per_reader, tt::tile_size(tensor_data_formats[t]));
        coalesced_page_sizes.push_back(coalesced_page_size);
        coalesced_num_pages.push_back(coalesced_num_page);
    }

    std::vector<uint32_t> bank_ids;
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
        uint32_t bank_id = core_index;
        uint32_t vc = (bank_id & 0x1) + 2;
        bank_ids.push_back(bank_id);

        // Compare with previous cores' vc
        for (size_t j = 0; j < core_index; ++j) {
            const CoreCoord& prev_core = reader_cores[j];
            if (prev_core.y == core.y and
                (((bank_id & 0x1) + 2) == ((bank_ids[j] & 0x1) + 2))) {  // same vc and same row
                vc = ((vc + 1) & 0x1) + 2;
                break;
            }
        }

        std::vector<uint32_t> reader_rt_args = {bank_id, vc, total_num_blocks_in_buffer};
        reader_rt_args.insert(reader_rt_args.end(), page_sizes.begin(), page_sizes.end());
        reader_rt_args.insert(reader_rt_args.end(), block_num_pages.begin(), block_num_pages.end());
        reader_rt_args.insert(reader_rt_args.end(), tensor_block_num_tiles.begin(), tensor_block_num_tiles.end());

        reader_desc.runtime_args.emplace_back(core, std::move(reader_rt_args));

        /* writer kernel */
        std::vector<uint32_t> writer_rt_args;
        writer_rt_args.insert(writer_rt_args.end(), coalesced_page_sizes.begin(), coalesced_page_sizes.end());
        writer_rt_args.insert(writer_rt_args.end(), coalesced_num_pages.begin(), coalesced_num_pages.end());
        writer_rt_args.insert(writer_rt_args.end(), tensor_block_num_tiles.begin(), tensor_block_num_tiles.end());
        writer_rt_args.insert(writer_rt_args.end(), tensor_tile_sizes.begin(), tensor_tile_sizes.end());
        for (auto tensor_shape : tensor_shapes) {  // block_height_in_itles
            writer_rt_args.push_back(tensor_shape[0] / num_blocks);
        }

        writer_desc.runtime_args.emplace_back(core, std::move(writer_rt_args));
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
