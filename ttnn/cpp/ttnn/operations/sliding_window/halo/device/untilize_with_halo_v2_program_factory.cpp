// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_halo_v2_program_factory.hpp"

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

// In order to make circular buffer indicies sequential, we use variable to keep track of the next available index.
// Circular buffer indices should be assigned right before their creation.
struct CBIndices {
    // Invalid value for cb id is 32, number greater than the maximum number of index circular buffer can have.
    // Not assigning get_next_cb_index() value before creating cb will throw exception in circular_buffer_types.cpp
    // which can be used as a reminder.
    uint32_t src_cb_id = 32;
    uint32_t pad_cb_id = 32;
    uint32_t out_cb_id = 32;

    // Additional CBs for sharded data kernel configs
    uint32_t padding_config = 32;
    uint32_t gather_config0 = 32;
    uint32_t gather_config1 = 32;
    uint32_t untilize_out_cb_id0 = 32;
    uint32_t untilize_out_cb_id1 = 32;
    uint32_t get_next_cb_id() { return next_cb_id++; }

private:
    uint32_t next_cb_id = tt::CBIndex::c_0;
};

operation::ProgramWithCallbacks untilize_with_halo_multi_core_v2(
    Program& program,
    const Tensor& input_tensor,
    const uint32_t pad_val,
    const uint32_t ncores_nhw,
    const uint32_t max_out_nsticks_per_core,
    const Tensor& padding_config,
    const Tensor& gather_config0,
    const Tensor& gather_config1,
    const std::vector<uint16_t>& number_of_blocks_per_core,
    const bool remote_read,
    const bool transpose_mcast,
    Tensor& output_tensor,
    const int block_size,
    const bool capture_buffers) {
    IDevice* device = input_tensor.device();
    Buffer* src_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const bool skip_untilize = input_tensor.get_layout() == Layout::ROW_MAJOR;

    const auto input_shape = input_tensor.get_padded_shape();
    const auto output_shape = output_tensor.get_padded_shape();

    const tt::DataFormat in_df = datatype_to_dataformat_converter(input_tensor.get_dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output_tensor.get_dtype());
    const uint32_t out_nbytes = datum_size(out_df);

    const CoreRangeSet all_cores = output_tensor.shard_spec().value().grid;
    const ShardOrientation shard_orientation = output_tensor.shard_spec().value().orientation;
    const auto input_shard_shape = output_tensor.shard_spec().value().shape;
    const auto output_shard_shape = output_tensor.shard_spec().value().shape;
    TT_ASSERT(input_shard_shape[1] == output_shard_shape[1], "Expected input and output shard shapes to match");
    const uint32_t input_nhw_height = input_shape[0] * input_shape[1] * input_shape[2];
    const uint32_t remapped_input_shard_shape_for_output_grid = tt::div_up(input_nhw_height, ncores_nhw);
    uint32_t ntiles_per_block = tt::div_up(input_shard_shape[1], TILE_WIDTH);
    uint32_t input_nblocks_per_core = tt::div_up(remapped_input_shard_shape_for_output_grid, TILE_HEIGHT);
    uint32_t input_npages = ntiles_per_block * input_nblocks_per_core;

    uint32_t in_page_size = tt::tt_metal::detail::TileSize(in_df);
    if (skip_untilize) {
        uint32_t in_nbytes = datum_size(in_df);
        in_page_size = input_shard_shape[1] * in_nbytes;
        input_npages = remapped_input_shard_shape_for_output_grid;
    }

    const uint32_t out_stick_nbytes = output_shard_shape[1] * out_nbytes;
    const uint32_t out_tile_size = tt::tt_metal::detail::TileSize(out_df);

    CBIndices cb_indices = CBIndices();

    // The input CB can either be tiled or row-major
    cb_indices.src_cb_id = cb_indices.get_next_cb_id();
    auto src_cb_config = CircularBufferConfig(input_npages * in_page_size, {{cb_indices.src_cb_id, in_df}})
                             .set_page_size(cb_indices.src_cb_id, in_page_size)
                             .set_globally_allocated_address(*src_buffer);
    auto src_cb = CreateCircularBuffer(program, all_cores, src_cb_config);
    log_debug(tt::LogOp, "CB {} :: npages = {}, pagesize = {}", cb_indices.src_cb_id, input_npages, in_page_size);

    // We need to clamp in the case that the block size is larger than the nhw input size
    TT_FATAL(block_size % TILE_HEIGHT == 0, "Block size must be a multiple of tile height (was {})", block_size);
    const uint32_t clamped_block_size_height =
        std::min(static_cast<uint32_t>(block_size), input_nblocks_per_core * TILE_HEIGHT);
    TT_FATAL(
        clamped_block_size_height % TILE_HEIGHT == 0,
        "Block size must be a multiple of tile height (was {})",
        clamped_block_size_height);

    uint32_t input_to_writer_cb_id0 = cb_indices.src_cb_id;
    uint32_t input_to_writer_cb_id1 = cb_indices.src_cb_id;
    if (!skip_untilize) {
        cb_indices.untilize_out_cb_id0 = cb_indices.get_next_cb_id();
        cb_indices.untilize_out_cb_id1 = cb_indices.get_next_cb_id();
        input_to_writer_cb_id0 = cb_indices.untilize_out_cb_id0;
        input_to_writer_cb_id1 = cb_indices.untilize_out_cb_id1;
        const uint32_t output_ntiles = (clamped_block_size_height / TILE_HEIGHT) * ntiles_per_block;
        auto untilize_out_cb_config0 =
            CircularBufferConfig(output_ntiles * out_tile_size, {{cb_indices.untilize_out_cb_id0, out_df}})
                .set_page_size(cb_indices.untilize_out_cb_id0, out_tile_size);
        auto untilize_out_cb0 = CreateCircularBuffer(program, all_cores, untilize_out_cb_config0);
        auto untilize_out_cb_config1 =
            CircularBufferConfig(output_ntiles * out_tile_size, {{cb_indices.untilize_out_cb_id1, out_df}})
                .set_page_size(cb_indices.untilize_out_cb_id1, out_tile_size);
        auto untilize_out_cb1 = CreateCircularBuffer(program, all_cores, untilize_out_cb_config1);
    }

    uint32_t out_cb_pagesize = out_stick_nbytes;
    uint32_t out_cb_npages = max_out_nsticks_per_core;
    cb_indices.out_cb_id = cb_indices.get_next_cb_id();
    auto out_cb_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{cb_indices.out_cb_id, out_df}})
                             .set_page_size(cb_indices.out_cb_id, out_cb_pagesize)
                             .set_globally_allocated_address(*dst_buffer);
    auto out_cb = CreateCircularBuffer(program, all_cores, out_cb_config);
    log_debug(tt::LogOp, "CB {} :: npages = {}, pagesize = {}", cb_indices.out_cb_id, out_cb_npages, out_cb_pagesize);

    // Used for storing padding immediate values (TODO: use zeroed memory region instead)
    uint32_t pad_cb_pagesize = out_stick_nbytes;
    uint32_t pad_cb_npages = 1;
    cb_indices.pad_cb_id = cb_indices.get_next_cb_id();
    auto pad_cb_config = CircularBufferConfig(pad_cb_pagesize * pad_cb_npages, {{cb_indices.pad_cb_id, out_df}})
                             .set_page_size(cb_indices.pad_cb_id, pad_cb_pagesize);
    auto pad_cb = CreateCircularBuffer(program, all_cores, pad_cb_config);
    log_debug(tt::LogOp, "CB {} :: npages = {}, pagesize = {}", cb_indices.pad_cb_id, pad_cb_npages, pad_cb_pagesize);

    tt::DataFormat kernel_config_df = tt::DataFormat::RawUInt16;  // NOTE: UInt16 is not supported for CB types
    uint32_t pagesize = 0;

    if (!skip_untilize) {
        const std::string compute_kernel_name =
            "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/compute/pack_untilize.cpp";
        const std::vector<uint32_t> compute_ct_args = {
            cb_indices.src_cb_id,
            input_to_writer_cb_id0,
            input_to_writer_cb_id1,
            ntiles_per_block,                        // number of tiles in the width dimension (channels)
            clamped_block_size_height / TILE_HEIGHT  // number of tiles in height dimension that make up a block

        };
        tt::log_info("CT args: {} - RT args: {}", compute_ct_args, number_of_blocks_per_core);
        KernelHandle untilize_kernel_id =
            CreateKernel(program, compute_kernel_name, all_cores, ComputeConfig{.compile_args = compute_ct_args});

        const bool is_rm_orientation = shard_orientation == ShardOrientation::ROW_MAJOR;
        const auto cores = corerange_to_cores(all_cores, std::nullopt, is_rm_orientation);
        for (int core_id = 0; core_id < cores.size(); core_id++) {
            SetRuntimeArgs(program, untilize_kernel_id, cores[core_id], {number_of_blocks_per_core[core_id]});
        }
    }

    TT_ASSERT(padding_config.get_dtype() == DataType::UINT16);
    TT_ASSERT(gather_config0.get_dtype() == DataType::UINT16);
    TT_ASSERT(gather_config1.get_dtype() == DataType::UINT16);

    auto padding_config_buffer = padding_config.device_buffer();
    const uint32_t num_cores = all_cores.num_cores();
    cb_indices.padding_config = cb_indices.get_next_cb_id();
    auto padding_config_cb_config =
        CircularBufferConfig(padding_config_buffer->size() / num_cores, {{cb_indices.padding_config, kernel_config_df}})
            .set_page_size(cb_indices.padding_config, padding_config_buffer->page_size())
            .set_globally_allocated_address(*padding_config_buffer);
    CBHandle padding_config_cb = CreateCircularBuffer(program, all_cores, padding_config_cb_config);

    auto gather_config_buffer0 = gather_config0.device_buffer();
    cb_indices.gather_config0 = cb_indices.get_next_cb_id();
    auto gather_cb_config0 =
        CircularBufferConfig(gather_config_buffer0->size() / num_cores, {{cb_indices.gather_config0, kernel_config_df}})
            .set_page_size(cb_indices.gather_config0, gather_config_buffer0->page_size())
            .set_globally_allocated_address(*gather_config_buffer0);
    CBHandle gather_config_cb0 = CreateCircularBuffer(program, all_cores, gather_cb_config0);

    auto gather_config_buffer1 = gather_config1.device_buffer();
    cb_indices.gather_config1 = cb_indices.get_next_cb_id();
    auto gather_cb_config1 =
        CircularBufferConfig(gather_config_buffer1->size() / num_cores, {{cb_indices.gather_config1, kernel_config_df}})
            .set_page_size(cb_indices.gather_config1, gather_config_buffer1->page_size())
            .set_globally_allocated_address(*gather_config_buffer1);
    CBHandle gather_config_cb1 = CreateCircularBuffer(program, all_cores, gather_cb_config1);

    const bool is_block_sharded = input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED;
    const bool is_width_sharded = input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED;

    auto aligned_input_nstick_nbytes = out_stick_nbytes;
    log_debug(tt::LogOp, "out_stick_nbytes = {}", out_stick_nbytes);
    log_debug(tt::LogOp, "input_tensor.buffer()->alignment() = {}", input_tensor.buffer()->alignment());

    if (out_stick_nbytes % input_tensor.buffer()->alignment() != 0) {
        aligned_input_nstick_nbytes = tt::round_up(out_stick_nbytes, input_tensor.buffer()->alignment());
    }

    const uint32_t block_stride = 2;  // Skip every 2nd block because of split reader
    const std::string reader_kernel_name =
        "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather.cpp";
    std::vector<uint32_t> reader_ct_args = {
        0,  // padding_config_cb_id
        0,  // gather_config_cb_id
        cb_indices.src_cb_id,
        input_to_writer_cb_id0,
        cb_indices.out_cb_id,
        cb_indices.pad_cb_id,
        pad_val,
        input_npages,
        out_stick_nbytes,
        is_block_sharded,
        remote_read,
        (uint32_t)(transpose_mcast ? 1 : 0),
        is_width_sharded,
        aligned_input_nstick_nbytes,
        skip_untilize,
        skip_untilize ? input_npages : clamped_block_size_height,
        ntiles_per_block,
        0,            // Block start offset
        block_stride  // Block stride
    };

    reader_ct_args[0] = 0;
    reader_ct_args[1] = cb_indices.gather_config0;
    KernelHandle reader_kernel_id0 = CreateKernel(
        program,
        reader_kernel_name,
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_ct_args});

    reader_ct_args[0] = cb_indices.padding_config;
    reader_ct_args[1] = cb_indices.gather_config1;
    reader_ct_args[3] = input_to_writer_cb_id1;
    reader_ct_args[17] = 1;  // block start offset
    KernelHandle reader_kernel_id1 = CreateKernel(
        program,
        reader_kernel_name,
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    // Capture padding_config_buffer, local_config_buffer, remote_config_buffer to cache this with the program
    if (!capture_buffers) {
        padding_config_buffer = nullptr;
        gather_config_buffer0 = nullptr;
        gather_config_buffer1 = nullptr;
    }
    auto override_runtime_arguments_callback = [src_cb,
                                                out_cb,
                                                padding_config_cb,
                                                gather_config_cb0,
                                                gather_cb_config1,
                                                padding_config_buffer,
                                                gather_config_buffer0,
                                                gather_config_buffer1](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, src_cb, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::data_movement::detail
