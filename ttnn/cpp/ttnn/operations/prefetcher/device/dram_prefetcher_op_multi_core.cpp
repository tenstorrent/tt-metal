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
    const std::vector<Tensor>& tensors,
    const std::optional<tt::tt_metal::v1::experimental::GlobalCircularBuffer>& global_cb,
    Tensor& output_tensor) {
    TT_FATAL(global_cb != std::nullopt, "Global circular buffer must be provided");

    Program program{};

    // In validate we make sure that all tensors are on the same device
    tt::tt_metal::Device* device = tensors[0].device();
    uint32_t num_tensors = tensors.size();

    // // WORKAROUND
    //
    //
    //

    // uint32_t global_cb_size = 750000;
    // uint32_t num_receivers_tmp = 2;
    // CoreCoord dram_reader_core_coord_tmp = CoreCoord{0, 0};
    // CoreRangeSet dram_reader_core{std::set<CoreRange>{CoreRange{dram_reader_core_coord_tmp}}};

    // // L1 receiver cores
    // CoreRange l1_receiver_core_coord_range = CoreRange(CoreCoord{0, 0});
    // if (device->arch() == tt::ARCH::GRAYSKULL) {
    //     l1_receiver_core_coord_range = CoreRange{CoreCoord{0, 1}, CoreCoord{0, num_receivers_tmp}};
    // } else {
    //     l1_receiver_core_coord_range = CoreRange{CoreCoord{1, 0}, CoreCoord{num_receivers_tmp, 0}};
    // }
    // CoreRangeSet l1_receiver_core{std::set<CoreRange>{l1_receiver_core_coord_range}};

    // std::unordered_map<CoreCoord, CoreRangeSet> sender_receiver_core_mapping;
    // sender_receiver_core_mapping[dram_reader_core_coord_tmp] = l1_receiver_core;

    // auto global_cb = tt::tt_metal::v1::experimental::CreateGlobalCircularBuffer(
    //     device, sender_receiver_core_mapping, global_cb_size, tt::tt_metal::BufferType::L1);
    //
    //
    //

    auto dram_reader_cores = global_cb->sender_cores();
    uint32_t num_receivers = global_cb->sender_receiver_core_mapping().begin()->second.size();

    // DRAM reader CB
    uint32_t in1_reader_cb_index = 0;
    // uint32_t in1_reader_cb_size = 750000;  // Total available L1 per core: 1.5 MB; we take half the L1, so 750000
    // bytes
    uint32_t in1_reader_cb_size = global_cb->size();
    tt::DataFormat in1_reader_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t in1_reader_cb_single_tile_size = 2048;

    tt_metal::CircularBufferConfig in1_reader_cb_config =
        tt_metal::CircularBufferConfig(in1_reader_cb_size, {{in1_reader_cb_index, in1_reader_cb_data_format}})
            .set_page_size(in1_reader_cb_index, in1_reader_cb_single_tile_size);
    auto in1_reader_cb = tt_metal::CreateCircularBuffer(program, dram_reader_cores, in1_reader_cb_config);

    // Writer CB maps inplace with global CB
    uint32_t in1_writer_cb_index = 31;
    uint32_t in1_writer_cb_size = global_cb->size();
    tt::DataFormat in1_writer_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t in1_writer_cb_single_tile_size = 2048;

    tt_metal::CircularBufferConfig in1_writer_cb_config =
        tt_metal::CircularBufferConfig(in1_writer_cb_size).set_globally_allocated_address(*output_tensor.buffer());
    in1_writer_cb_config.remote_index(in1_writer_cb_index)
        .set_page_size(in1_writer_cb_single_tile_size)
        .set_data_format(in1_writer_cb_data_format);
    auto in1_writer_cb =
        tt_metal::v1::experimental::CreateCircularBuffer(program, dram_reader_cores, in1_writer_cb_config, *global_cb);

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
        dram_reader_cores,
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
        dram_reader_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = in1_writer_compile_time_args});

    // reader rt
    uint32_t total_num_blocks_in_buffer = 3;  // TODO: how big should reader CB be? here it's triple buffered

    uint32_t bank_start_id = 1;
    std::vector<uint32_t> bank_ids;

    // Store all cores in a vector for easier access to previous cores
    std::vector<CoreCoord> all_cores;
    for (const auto& range : dram_reader_cores.ranges()) {
        for (const auto& core : range) {
            all_cores.push_back(core);
        }
    }

    for (int core_index = 0; core_index < all_cores.size(); core_index++) {
        const auto& core = all_cores[core_index];
        uint32_t bank_id = core_index + bank_start_id;
        uint32_t vc = bank_id & 0x1;

        bank_ids.push_back(bank_id);

        // Compare with previous cores
        for (size_t j = 0; j < core_index; ++j) {
            const CoreCoord& prev_core = all_cores[j];
            if (prev_core.y == core.y and ((bank_id & 0x1) == (bank_ids[j] & 0x1))) {  // same vc and same row
                vc = (vc + 1) & 0x1;
                break;
            }
        }

        log_info("core: {}, vc: {}, bank_id: {}", core, vc, bank_id);

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
        tt_metal::SetRuntimeArgs(program, in1_reader_kernel, core, reader_rt_args);

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
        tt_metal::SetRuntimeArgs(program, in1_writer_kernel, core, writer_rt_args);
    }

    auto override_runtime_arguments_callback = [in1_reader_kernel, in1_writer_kernel, dram_reader_cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        for (const auto& range : dram_reader_cores.ranges()) {
            for (const auto& core_coord : range) {
                // TODO: set runtime args for reader and writer
                auto& reader_runtime_args = GetRuntimeArgs(program, in1_reader_kernel, core_coord);
                auto& writer_runtime_args = GetRuntimeArgs(program, in1_writer_kernel, core_coord);
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::dram_prefetcher
