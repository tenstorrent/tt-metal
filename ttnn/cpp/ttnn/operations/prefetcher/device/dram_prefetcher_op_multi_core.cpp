// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dram_prefetcher_op.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

namespace ttnn::operations::dram_prefetcher {

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
    const std::vector<Tensor>& tensors, uint32_t global_cb_addr, uint32_t global_cb_size, uint32_t num_receivers) {
    Program program{};

    tt::tt_metal::Device* device = tensors[0].device();

    uint32_t num_tensors = tensors.size();
    uint32_t start_tile_id = 0;

    // TODO: handle multiple tensors

    uint32_t kt = tensors[0].get_legacy_shape()[0] / 32;
    uint32_t nt = tensors[0].get_legacy_shape()[1] / 32;

    uint32_t num_blocks = 24;  // number of receiver cores
    uint32_t in1_block_h = kt / num_blocks;
    uint32_t in1_block_w = nt;
    uint32_t in1_block_num_tiles = in1_block_h * in1_block_w;
    uint32_t in1_num_tile_rows_write = in1_block_h;

    tt::DataFormat input_tensor_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensors[0].get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(input_tensor_data_format);

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

    // DRAM reader CB
    uint32_t in1_reader_cb_index = 0;
    uint32_t in1_reader_cb_size = in1_block_h * in1_block_w * single_tile_size * 3;

    uint32_t in1_reader_page_sizes[num_tensors], in1_reader_num_pages[num_tensors];
    for (uint32_t i = 0; i < num_tensors; ++i) {
        get_max_page_size_and_num_pages(
            in1_block_num_tiles, single_tile_size, in1_reader_page_sizes[i], in1_reader_num_pages[i]);
    }

    log_info("in1_reader_page_sizes[0]: {}", in1_reader_page_sizes[0]);
    log_info("in1_reader_num_pages[0]: {}", in1_reader_num_pages[0]);

    tt_metal::CircularBufferConfig in1_reader_cb_config =
        tt_metal::CircularBufferConfig(in1_reader_cb_size, {{in1_reader_cb_index, input_tensor_data_format}})
            .set_page_size(in1_reader_cb_index, single_tile_size);
    auto in1_reader_cb = tt_metal::CreateCircularBuffer(program, dram_reader_core, in1_reader_cb_config);

    uint32_t in1_receiver_block_num_tile = in1_block_h * in1_block_w / num_receivers;
    uint32_t in1_writer_page_sizes[num_tensors], in1_writer_num_pages[num_tensors];
    for (uint32_t i = 0; i < num_tensors; ++i) {
        get_max_page_size_and_num_pages(
            in1_block_w / num_receivers, single_tile_size, in1_writer_page_sizes[i], in1_writer_num_pages[i]);
    }

    log_info("in1_writer_page_sizes[0]: {}", in1_writer_page_sizes[0]);
    log_info("in1_writer_num_pages[0]: {}", in1_writer_num_pages[0]);

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
        (std::uint32_t)global_cb_addr,
        (std::uint32_t)global_cb_size,
        (std::uint32_t)num_receivers,
        (std::uint32_t)num_tensors};

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
    auto dram_reader_core_coord = dram_reader_core.ranges().begin()->start_coord;
    log_info("dram_reader_core_coord: {}", dram_reader_core_coord);
    auto dram_reader_core_coord_physical = device->worker_core_from_logical_core(dram_reader_core_coord);
    uint32_t bank_id = 0;
    uint32_t vc = bank_id & 0x1;
    std::vector<uint32_t> reader_rt_args = {(std::uint32_t)bank_id, (std::uint32_t)vc};
    // tensor addresses
    for (uint32_t i = 0; i < num_tensors; ++i) {
        reader_rt_args.push_back(tensors[i]->address());
    }
    // page size
    for (uint32_t i = 0; i < num_tensors; ++i) {
        reader_rt_args.push_back(in1_reader_page_sizes[i]);
    }
    // num pages
    for (uint32_t i = 0; i < num_tensors; ++i) {
        reader_rt_args.push_back(in1_reader_num_pages[i]);
    }
    // num blocks
    for (uint32_t i = 0; i < num_tensors; ++i) {
        reader_rt_args.push_back(num_blocks);
    }
    // num tiles
    for (uint32_t i = 0; i < num_tensors; ++i) {
        reader_rt_args.push_back(in1_block_num_tiles);
    }
    tt_metal::SetRuntimeArgs(program, in1_reader_kernel, dram_reader_core_coord, reader_rt_args);

    // in1 writer rt
    std::vector<CoreCoord> l1_receiver_core_coords;
    for (auto l1_receiver_core_coord : *l1_receiver_cores.ranges().begin()) {
        l1_receiver_core_coords.push_back(l1_receiver_core_coord);
    }
    std::vector<uint32_t> writer_rt_args;
    // receiver core coords x
    for (uint32_t i = 0; i < num_receivers; ++i) {
        auto l1_receiver_core_coord_physical = device->worker_core_from_logical_core(l1_receiver_core_coords[i]);
        writer_rt_args.push_back(l1_receiver_core_coord_physical.x);
    }
    // receiver core coords y
    for (uint32_t i = 0; i < num_receivers; ++i) {
        auto l1_receiver_core_coord_physical = device->worker_core_from_logical_core(l1_receiver_core_coords[i]);
        writer_rt_args.push_back(l1_receiver_core_coord_physical.y);
    }
    // page size
    for (uint32_t i = 0; i < num_tensors; ++i) {
        writer_rt_args.push_back(in1_writer_page_size);
    }
    // num pages
    for (uint32_t i = 0; i < num_tensors; ++i) {
        writer_rt_args.push_back(in1_writer_num_pages);
    }
    // num blocks
    for (uint32_t i = 0; i < num_tensors; ++i) {
        writer_rt_args.push_back(num_blocks);
    }
    // num tiles
    for (uint32_t i = 0; i < num_tensors; ++i) {
        writer_rt_args.push_back(in1_block_num_tiles);
    }
    // single tile size
    for (uint32_t i = 0; i < num_tensors; ++i) {
        writer_rt_args.push_back(single_tile_size);
    }
    // num tile rows write
    for (uint32_t i = 0; i < num_tensors; ++i) {
        writer_rt_args.push_back(in1_num_tile_rows_write);
    }
    tt_metal::SetRuntimeArgs(program, in1_writer_kernel, dram_reader_core_coord, writer_rt_args);

    auto override_runtime_arguments_callback = (){};

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::dram_prefetcher
