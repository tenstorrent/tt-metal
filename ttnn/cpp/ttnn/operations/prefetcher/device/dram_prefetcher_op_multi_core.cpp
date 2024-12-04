// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dram_prefetcher_op.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include "tt_metal/impl/buffers/global_circular_buffer.hpp"
#include "tt_metal/include/tt_metal/global_circular_buffer.hpp"

namespace ttnn::operations::dram_prefetcher {

using std::vector;
using namespace tt::constants;

void get_max_page_size_and_num_pages(
    uint32_t num_tiles, uint32_t num_datums_per_tile, uint32_t& page_size, uint32_t& num_pages) {
    uint64_t total_size = static_cast<uint64_t>(num_tiles) * num_datums_per_tile;

    page_size = (8192 / num_datums_per_tile) * num_datums_per_tile;
    while (total_size % page_size != 0 && page_size >= num_datums_per_tile) {
        page_size -= num_datums_per_tile;
    }
    num_pages = total_size / page_size;
}

operation::ProgramWithCallbacks dram_prefetcher_multi_core(
    std::vector<Tensor>& tensors,
    // std::shared_ptr<tt_metal::v1::experimental::GlobalCircularBuffer> global_cb,
    uint32_t num_receivers) {
    Program program{};

    // In validate we make sure that all tensors are on the same device
    tt::tt_metal::Device* device = tensors[0].device();

    uint32_t num_tensors = tensors.size();

    // // DRAM reader cores
    // CoreCoord dram_reader_core_coord = CoreCoord{0, 0};
    // CoreRangeSet dram_reader_core{std::set<CoreRange>{CoreRange{dram_reader_core_coord}}};

    // // L1 receiver cores
    // CoreRange l1_receiver_core_coord_range = CoreRange(CoreCoord{0, 0});
    // if (device->arch() == tt::ARCH::GRAYSKULL) {
    //     l1_receiver_core_coord_range = CoreRange{CoreCoord{0, 1}, CoreCoord{0, num_receivers}};
    // } else {
    //     l1_receiver_core_coord_range = CoreRange{CoreCoord{1, 0}, CoreCoord{num_receivers, 0}};
    // }
    // CoreRangeSet l1_receiver_core{std::set<CoreRange>{l1_receiver_core_coord_range}};

    // std::unordered_map<CoreCoord, CoreRangeSet> sender_receiver_core_mapping;
    // sender_receiver_core_mapping[dram_reader_core_coord] = l1_receiver_core;

    uint32_t global_cb_size = 750000;

    // DRAM reader cores
    CoreCoord dram_reader_core_coord = CoreCoord{0, 0};
    CoreRangeSet dram_reader_core{std::set<CoreRange>{CoreRange{dram_reader_core_coord}}};

    // L1 receiver cores
    CoreRange l1_receiver_core_coord_range = CoreRange(CoreCoord{0, 0});
    if (device->arch() == tt::ARCH::GRAYSKULL) {
        l1_receiver_core_coord_range = CoreRange{CoreCoord{0, 1}, CoreCoord{0, num_receivers}};
    } else {
        l1_receiver_core_coord_range = CoreRange{CoreCoord{1, 0}, CoreCoord{num_receivers, 0}};
    }
    CoreRangeSet l1_receiver_core{std::set<CoreRange>{l1_receiver_core_coord_range}};

    std::unordered_map<CoreCoord, CoreRangeSet> sender_receiver_core_mapping;
    sender_receiver_core_mapping[dram_reader_core_coord] = l1_receiver_core;

    auto global_cb = tt_metal::v1::experimental::CreateGlobalCircularBuffer(
        device, sender_receiver_core_mapping, global_cb_size, tt_metal::BufferType::L1);

    // // Get l1_receiver_core from global_cb
    // std::set<CoreRange> l1_sender_cores;
    // std::set<CoreRange> l1_receiver_cores;
    // for (const auto sender_core : global_cb->sender_cores()) {
    //     l1_sender_cores.push_back(sender_core);
    // }
    // for (const auto receiver_core : global_cb->receiver_cores()) {
    //     l1_receiver_cores.push_back(receiver_core);
    // }

    // DRAM reader CB
    uint32_t in1_reader_cb_index = 0;
    uint32_t in1_reader_cb_size = 750000;  // Total available L1 per core: 1.5 MB; we take half the L1, so 750000 bytes
    tt::DataFormat in1_reader_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t in1_reader_cb_single_tile_size = 2048;

    tt_metal::CircularBufferConfig in1_reader_cb_config =
        tt_metal::CircularBufferConfig(in1_reader_cb_size, {{in1_reader_cb_index, in1_reader_cb_data_format}})
            .set_page_size(in1_reader_cb_index, in1_reader_cb_single_tile_size);
    auto in1_reader_cb = tt_metal::CreateCircularBuffer(program, dram_reader_core, in1_reader_cb_config);

    // Writer CB maps inplace with global CB
    uint32_t in1_writer_cb_index = 31;
    uint32_t in1_writer_cb_size = global_cb->size();
    tt::DataFormat in1_writer_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t in1_writer_cb_single_tile_size = 2048;

    tt_metal::CircularBufferConfig in1_writer_cb_config = tt_metal::CircularBufferConfig(in1_writer_cb_size);
    in1_writer_cb_config.remote_index(in1_writer_cb_index)
        .set_page_size(in1_writer_cb_single_tile_size)
        .set_data_format(in1_writer_cb_data_format);
    auto in1_writer_cb =
        tt_metal::v1::experimental::CreateCircularBuffer(program, dram_reader_core, in1_writer_cb_config, *global_cb);

    // Set up per tensor
    uint32_t in1_writer_page_sizes[num_tensors], in1_writer_num_pages[num_tensors];
    uint32_t in1_reader_page_sizes[num_tensors], in1_reader_num_pages[num_tensors], single_tile_sizes[num_tensors];
    uint32_t in1_block_num_tiles[num_tensors], in1_num_tile_rows_write[num_tensors];
    uint32_t num_blocks = 24;  // number of receiver cores
    for (uint32_t i = 0; i < num_tensors; ++i) {
        uint32_t kt = tensors[i].get_legacy_shape()[0] / 32;
        uint32_t nt = tensors[i].get_legacy_shape()[1] / 32;

        uint32_t in1_block_h = kt / num_blocks;
        uint32_t in1_block_w = nt;
        in1_block_num_tiles[i] = in1_block_h * in1_block_w;
        in1_num_tile_rows_write[i] = in1_block_h;

        tt::DataFormat input_tensor_data_format =
            tt::tt_metal::datatype_to_dataformat_converter(tensors[i].get_dtype());
        single_tile_sizes[i] = tt::tt_metal::detail::TileSize(input_tensor_data_format);

        get_max_page_size_and_num_pages(
            in1_block_num_tiles[i], single_tile_sizes[i], in1_reader_page_sizes[i], in1_reader_num_pages[i]);

        log_info("in1_reader_page_sizes[{}]: {}", i, in1_reader_page_sizes[i]);
        log_info("in1_reader_num_pages[{}]: {}", i, in1_reader_num_pages[i]);

        get_max_page_size_and_num_pages(
            in1_block_w / num_receivers, single_tile_sizes[i], in1_writer_page_sizes[i], in1_writer_num_pages[i]);

        log_info("in1_writer_page_sizes[{}]: {}", i, in1_writer_page_sizes[i]);
        log_info("in1_writer_num_pages[{}]: {}", i, in1_writer_num_pages[i]);
    }

    // in1 reader
    std::vector<uint32_t> in1_reader_compile_time_args = {
        (std::uint32_t)tt_metal::NOC::RISCV_0_default, (std::uint32_t)num_tensors};

    auto in1_reader_kernel = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/prefetcher/device/kernels/reader_dram.cpp",
        dram_reader_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = in1_reader_compile_time_args});

    // in1 writer
    std::vector<uint32_t> in1_writer_compile_time_args = {
        (std::uint32_t)tt_metal::NOC::RISCV_0_default,
        (std::uint32_t)num_blocks,
        (std::uint32_t)num_receivers,
        (std::uint32_t)num_tensors,
        (std::uint32_t)in1_reader_cb_index,
        (std::uint32_t)in1_writer_cb_index};

    auto in1_writer_kernel = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/prefetcher/device/kernels/writer_l1.cpp",
        dram_reader_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = in1_writer_compile_time_args});

    // reader rt
    // auto dram_reader_core_coord = dram_reader_core.ranges().begin()->start_coord;
    // log_info("dram_reader_core_coord: {}", dram_reader_core_coord);
    // auto dram_reader_core_coord_physical = device->worker_core_from_logical_core(dram_reader_core_coord);
    uint32_t bank_id = 0;
    uint32_t vc = bank_id & 0x1;
    uint32_t total_num_blocks_in_buffer = 3;  // TODO: how big should reader CB be? here it's triple buffered
    std::vector<uint32_t> reader_rt_args = {
        (std::uint32_t)bank_id,
        (std::uint32_t)vc,
        (std::uint32_t)in1_reader_cb_size,
        (std::uint32_t)total_num_blocks_in_buffer,
        (std::uint32_t)num_blocks};
    // tensor addresses
    for (uint32_t i = 0; i < num_tensors; ++i) {
        reader_rt_args.push_back(tensors[i].buffer()->address());
    }
    // page size
    for (uint32_t i = 0; i < num_tensors; ++i) {
        reader_rt_args.push_back(in1_reader_page_sizes[i]);
    }
    // num pages
    for (uint32_t i = 0; i < num_tensors; ++i) {
        reader_rt_args.push_back(in1_reader_num_pages[i]);
    }
    // num tiles in block
    for (uint32_t i = 0; i < num_tensors; ++i) {
        reader_rt_args.push_back(in1_block_num_tiles[i]);
    }
    tt_metal::SetRuntimeArgs(program, in1_reader_kernel, dram_reader_core_coord, reader_rt_args);

    // in1 writer rt
    std::vector<uint32_t> writer_rt_args = {};
    // page size
    for (uint32_t i = 0; i < num_tensors; ++i) {
        writer_rt_args.push_back(in1_writer_page_sizes[i]);
    }
    // num pages
    for (uint32_t i = 0; i < num_tensors; ++i) {
        writer_rt_args.push_back(in1_writer_num_pages[i]);
    }
    // block num tiles
    for (uint32_t i = 0; i < num_tensors; ++i) {
        writer_rt_args.push_back(in1_block_num_tiles[i]);
    }
    // single tile size
    for (uint32_t i = 0; i < num_tensors; ++i) {
        writer_rt_args.push_back(single_tile_sizes[i]);
    }
    // num tile rows write
    for (uint32_t i = 0; i < num_tensors; ++i) {
        writer_rt_args.push_back(in1_num_tile_rows_write[i]);
    }
    tt_metal::SetRuntimeArgs(program, in1_writer_kernel, dram_reader_core_coord, writer_rt_args);

    auto override_runtime_arguments_callback = []() {};

    // return {.program = std::move(program)};
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::dram_prefetcher
